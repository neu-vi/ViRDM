from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lmdb
import numpy as np
from omegaconf import OmegaConf

from scripts.build_inference_config import (
    BASE_TIMESTEPS,
    RECIPES,
    render_inference_config,
    validate_inference_config,
)
from scripts.build_recipe import ROLLOUTS, render_recipe, validate
from scripts.check_artifacts import require_prompt_lmdb
from scripts.export_videos import load_generator_state
from scripts.package_checkpoint import package_checkpoint
from scripts.prepare_vbench_prompts import unique_rows
from scripts.sanitize_reference import REFERENCE_REVISION, sanitize_bundle
from scripts.summarize_vbench import NORMALIZATION, aggregate
from pipeline.causal_inference import _optional_step_tensor
from utils.dataset import CleanLatentLMDBDataset, PromptLMDBDataset, TextDataset
from virdm_integration.joint_text import prompt_rows_sha256
from virdm_integration.reference_builder import (
    build_nystrom_reference,
    median_bandwidth,
)
from trainer.virdm import Trainer, _RolloutChunk


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "config_virdm_bs64_1x8.yaml"
MANIFEST = REPO_ROOT / "artifacts" / "manifest.json"


class RecipeContractTest(unittest.TestCase):
    def test_wandb_logging_uses_the_public_metric_whitelist(self):
        trainer = Trainer.__new__(Trainer)
        trainer.is_main_process = True
        trainer.disable_wandb = False
        trainer.step = 7
        generator_logs = {
            "raw_mmd": 1.0,
            "k_gg": 2.0,
            "k_gr_nystrom": 3.0,
            "k_rr": 4.0,
            "generator_grad_norm": 5.0,
            "dynamic_reg/raw_loss": 6.0,
            "dynamic_reg/weighted_loss": 7.0,
            "dynamic_reg/s10_mean_px": 8.0,
            "dynamic_reg/hit_count_mean": 9.0,
            "dynamic_reg/pass_fraction": 10.0,
            "dynamic_reg/pair_score_mean_px": 11.0,
            "gpu/peak_allocated_gib": 12.0,
            "gpu/peak_reserved_gib": 13.0,
            "rollout_video": object(),
            "normalized_mmd": 99.0,
            "timing/generator_update_seconds": 99.0,
        }
        with patch("trainer.virdm.wandb.log") as wandb_log:
            trainer._log_step(generator_logs)
        logged = wandb_log.call_args.args[0]
        self.assertEqual(wandb_log.call_args.kwargs, {"step": 7})
        self.assertEqual(logged["step"], 7)
        self.assertIn("generator/raw_mmd", logged)
        self.assertIn("generator/dynamic_reg/pass_fraction", logged)
        self.assertIn("rollout/rollout_video", logged)
        self.assertNotIn("generator/rollout_video", logged)
        self.assertNotIn("generator/normalized_mmd", logged)
        self.assertNotIn("generator/timing/generator_update_seconds", logged)
        self.assertEqual(len(logged), 15)

    def test_eight_training_recipes(self):
        expected_steps = {
            "chunk4": BASE_TIMESTEPS,
            "chunk4+2": BASE_TIMESTEPS[:2],
            "chunk4+1": BASE_TIMESTEPS[:1],
            "bid4": BASE_TIMESTEPS,
        }
        for rollout in ROLLOUTS:
            for dynamic in (False, True):
                with self.subTest(rollout=rollout, dynamic=dynamic):
                    config = render_recipe(BASE_CONFIG, rollout, dynamic)
                    validate(config, rollout, dynamic)
                    self.assertEqual(list(config.denoising_step_list), expected_steps[rollout])
                    self.assertEqual(int(config.max_steps), 20)
                    self.assertEqual(int(config.log_iters), 20)
                    if rollout == "chunk4":
                        self.assertEqual(
                            list(config.virdm_first_frame_denoising_step_list), []
                        )
                        self.assertFalse(
                            bool(config.virdm_replay_first_frame_trajectory_noise)
                        )
                    elif rollout.startswith("chunk"):
                        self.assertEqual(
                            list(config.virdm_first_frame_denoising_step_list),
                            BASE_TIMESTEPS,
                        )
                        self.assertTrue(
                            bool(config.virdm_replay_first_frame_trajectory_noise)
                        )

    def test_single_gpu_recipes_preserve_global_population(self):
        for rollout in ROLLOUTS:
            for dynamic in (False, True):
                with self.subTest(rollout=rollout, dynamic=dynamic):
                    config = render_recipe(
                        BASE_CONFIG, rollout, dynamic, world_size=1
                    )
                    validate(config, rollout, dynamic, world_size=1)
                    self.assertEqual(int(config.virdm_grad_accum_steps), 64)
                    self.assertEqual(int(config.virdm_expected_world_size), 1)
                    self.assertEqual(int(config.virdm_expected_local_world_size), 1)
                    self.assertEqual(int(config.virdm_expected_global_rows), 64)
                    Trainer._validate_static_config(config)

    def test_four_inference_recipes(self):
        for recipe, spec in RECIPES.items():
            with self.subTest(recipe=recipe):
                config = render_inference_config(BASE_CONFIG, recipe, 21)
                validate_inference_config(config, recipe, 21)
                self.assertEqual(int(config.num_frame_per_block), spec["block"])
                self.assertEqual(
                    len(config.denoising_step_list), spec["later_steps"]
                )

    def test_causal_frame_divisibility(self):
        with self.assertRaises(ValueError):
            render_inference_config(BASE_CONFIG, "causal4", 20)
        with self.assertRaises(ValueError):
            render_inference_config(BASE_CONFIG, "bid4", 18)

    def test_empty_first_block_schedule_is_disabled(self):
        self.assertIsNone(_optional_step_tensor([]))
        tensor = _optional_step_tensor([1000, 750])
        self.assertEqual(tensor.tolist(), [1000, 750])

    def test_native_sdpa_preserves_padding_contract(self):
        import torch

        from wan.modules.attention import _scaled_dot_product_attention

        query = torch.zeros(1, 3, 1, 4)
        key = torch.zeros_like(query)
        value = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0]], [[3.0, 4.0, 5.0, 6.0]], [[99.0] * 4]]]
        )
        output = _scaled_dot_product_attention(
            query,
            key,
            value,
            q_lens=torch.tensor([2]),
            k_lens=torch.tensor([2]),
        )
        expected = torch.tensor(
            [[[[2.0, 3.0, 4.0, 5.0]], [[2.0, 3.0, 4.0, 5.0]], [[0.0] * 4]]]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_base_config_release_values(self):
        import torch

        config = OmegaConf.load(BASE_CONFIG)
        self.assertEqual(config.trainer, "virdm")
        self.assertEqual(int(config.seed), 0)
        self.assertEqual(float(config.lr), 2e-6)
        self.assertEqual(config.virdm_attention_backend, "flash_attn_2")
        self.assertEqual(config.virdm_flash_attn_version, "2.8.3.post1")
        self.assertTrue(bool(config.virdm_require_flash_attention))
        self.assertEqual(int(config.virdm_expected_global_rows), 64)
        self.assertEqual(float(config.virdm_joint_bandwidth_scale), 1.0)
        self.assertEqual(config.virdm_reference_revision, REFERENCE_REVISION)
        self.assertIsInstance(
            Trainer.__dict__["_validate_static_config"], staticmethod
        )
        self.assertIsInstance(
            Trainer.__dict__["_dynamic_reg_top5_flow_mean"], staticmethod
        )
        self.assertTrue(
            inspect.getsource(Trainer._full_all_chunks_rollout_video)
            .lstrip()
            .startswith("@torch.no_grad()")
        )
        self.assertTrue(
            inspect.getsource(Trainer._latent_features)
            .lstrip()
            .startswith("@torch.no_grad()")
        )
        flow = torch.zeros(2, 2, 10, 10)
        scores = Trainer.__new__(Trainer)._dynamic_reg_top5_flow_mean(flow)
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))
        trainer_source = (REPO_ROOT / "trainer" / "virdm.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("self.model.vae", trainer_source)
        train_source = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn("dist.destroy_process_group()", train_source)
        inference_source = (REPO_ROOT / "scripts" / "export_videos.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("require_flash_attention_2", inference_source)
        self.assertIn("ATTENTION_BACKEND", inference_source)
        attention_source = (REPO_ROOT / "wan" / "modules" / "attention.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("version=2,", attention_source)
        self.assertIn("fa_version=2,", attention_source)
        infer_launcher = (REPO_ROOT / "scripts" / "infer.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('-m torch.distributed.run', infer_launcher)
        for setup_name in ("setup_env.sh", "setup_vbench_env.sh"):
            setup_source = (REPO_ROOT / "scripts" / setup_name).read_text(
                encoding="utf-8"
            )
            self.assertIn("require_conda_python", setup_source)
        vbench_setup = (REPO_ROOT / "scripts" / "setup_vbench_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'pip install --no-build-isolation "${VBENCH_ROOT}"', vbench_setup
        )
        self.assertIn("'pip==24.0'", vbench_setup)
        self.assertIn("'setuptools==80.9.0'", vbench_setup)
        self.assertIn('-m pip check', vbench_setup)
        self.assertIn(
            "pip install --no-build-isolation \\\n"
            "  'detectron2@git+https://github.com/facebookresearch/detectron2.git@",
            vbench_setup,
        )
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("flash-attn==2.8.3.post1", requirements)
        self.assertIn("accelerate==1.3.0", requirements)
        self.assertIn("open_clip_torch==3.3.0", requirements)
        setup_source = (REPO_ROOT / "scripts" / "setup_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("require_flash_attention_2", setup_source)
        self.assertIn('Path(sysconfig.get_path("include"))', setup_source)
        self.assertIn('directory / "Python.h"', setup_source)
        self.assertIn("FLASH_ATTENTION_FORCE_BUILD=TRUE", setup_source)
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("moviepy==1.0.3", requirements)
        packager = (REPO_ROOT / "scripts" / "package_training_assets.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--dynamic-checkpoint", packager)
        self.assertIn("virdm_release_assets_v1", packager)
        self.assertNotIn("does **not** contain", packager)
        self.assertIn(
            'train.sh" 1 "$@"',
            (REPO_ROOT / "scripts" / "train_1gpu.sh").read_text(encoding="utf-8"),
        )
        self.assertIn(
            'train.sh" 8 "$@"',
            (REPO_ROOT / "scripts" / "train_1x8.sh").read_text(encoding="utf-8"),
        )
        reference_builder = (REPO_ROOT / "scripts" / "build_reference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("encode_text_table", reference_builder)
        self.assertIn("beta = float(visual_sigma) / float(text_sigma)", reference_builder)
        self.assertNotIn("from rdm", reference_builder)

    def test_rollout_state_is_constructible(self):
        marker = object()
        chunk = _RolloutChunk(
            noise=marker,
            context_noise=marker,
            first_frame_exit_index=0,
            first_frame_trajectory_noise=marker,
            block_exit_indices=marker,
            block_trajectory_noise=marker,
            conditional={},
            latent=marker,
            dataset_index=3,
            prompt="test",
        )
        self.assertEqual(chunk.dataset_index, 3)

    def test_reference_metadata_sanitizer_preserves_payload(self):
        import torch

        tensor = torch.arange(6).reshape(2, 3)
        bundle = {
            "Z": tensor,
            "alpha": torch.ones(2),
            "metadata": {
                "format": "one_forcing_internal",
                "source_lmdb": "/" + "mnt/localssd/private/data.mdb",
                "encoder_checkpoint": "/" + "home/user/checkpoint.pt",
                "rdm_source_revision": "unpublished",
                "sigma_scale": 1.0,
            },
        }
        sanitized = sanitize_bundle(bundle)
        self.assertTrue(torch.equal(sanitized["Z"], tensor))
        self.assertTrue(torch.equal(sanitized["alpha"], bundle["alpha"]))
        metadata = sanitized["metadata"]
        self.assertEqual(metadata["virdm_reference_revision"], REFERENCE_REVISION)
        self.assertNotIn("source_lmdb", metadata)
        self.assertNotIn("encoder_checkpoint", metadata)
        self.assertNotIn("rdm_source_revision", metadata)


class ArtifactContractTest(unittest.TestCase):
    def test_demo_prompt_suite(self):
        path = REPO_ROOT / "prompts" / "demos.txt"
        rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 100)
        self.assertTrue(all(row.strip() for row in rows))
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "7c52c098260ec3c7435db6eeada499ab5ef47488c7d0cad424ec65db4c9bcd5d",
        )

    def test_vbench_prompt_suite(self):
        expected = {
            "all_dimension.txt": (
                946,
                "f9b50654f81ab732b9235d5bb91f0c3a4d152fac19249ce3a3a9185b2d80d8c3",
            ),
            "all_dimension_extended.txt": (
                946,
                "c78f1ad32974503efaa8cc935347ace88f3a8b571adde6687bc8425bbf417cf3",
            ),
        }
        for name, (row_count, digest) in expected.items():
            with self.subTest(name=name):
                path = REPO_ROOT / "prompts" / "vbench" / name
                rows = path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(rows), row_count)
                self.assertTrue(all(row.strip() for row in rows))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_manifest_is_complete(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["format"], "virdm_artifacts_v1")
        official = manifest["official"]
        self.assertEqual(len(official["wan_2_1_t2v_1_3b"]["revision"]), 40)
        self.assertEqual(len(official["causal_ode_chunkwise"]["sha256"]), 64)
        for name in ("vjepa_2_1_vitl_384", "taew_2_1", "dense_flow"):
            self.assertEqual(len(official[name]["sha256"]), 64)
        siglip = official["siglip_2_so400m_256"]
        self.assertEqual(siglip["repo_id"], "timm/ViT-SO400M-16-SigLIP2-256")
        self.assertEqual(len(siglip["revision"]), 40)
        self.assertEqual(len(siglip["file_sha256"]), 4)
        for digest in siglip["file_sha256"].values():
            self.assertEqual(len(digest), 64)
        training = manifest["virdm_training_assets"]
        self.assertEqual(training["repo_id"], "cr8br0ze/ViRDM")
        self.assertEqual(training["repo_type"], "model")
        self.assertEqual(len(training["revision"]), 40)
        self.assertEqual(int(training["expected_prompt_rows"]), 6505)
        self.assertEqual(len(training["prompt_rows_sha256"]), 64)
        checkpoints = manifest["virdm_checkpoints"]
        self.assertEqual(checkpoints["repo_id"], "cr8br0ze/ViRDM")
        self.assertEqual(checkpoints["revision"], training["revision"])
        self.assertEqual(len(checkpoints["files"]), 2)
        for item in checkpoints["files"].values():
            self.assertEqual(len(item["sha256"]), 64)
            self.assertTrue(item["target"].endswith(".pt"))

    def test_prompt_only_lmdb(self):
        prompts = ["first prompt", "second prompt"]
        with tempfile.TemporaryDirectory() as directory:
            env = lmdb.open(directory, map_size=1 << 20, subdir=True)
            with env.begin(write=True) as txn:
                txn.put(b"prompts_shape", b"2")
                for index, prompt in enumerate(prompts):
                    txn.put(
                        f"prompts_{index}_data".encode(), prompt.encode("utf-8")
                    )
            env.sync()
            env.close()

            dataset = PromptLMDBDataset(directory)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["prompts"], prompts[1])
            dataset.env.close()
            require_prompt_lmdb(
                directory,
                expected_rows=2,
                expected_rows_sha256=prompt_rows_sha256(prompts),
            )

    def test_clean_latent_lmdb_uses_last_trajectory_row(self):
        import torch

        shape = (2, 21, 16, 60, 104)
        row = np.zeros(shape, dtype=np.float16)
        row[-1].fill(3.0)
        with tempfile.TemporaryDirectory() as directory:
            env = lmdb.open(directory, map_size=32 << 20, subdir=True)
            with env.begin(write=True) as txn:
                txn.put(b"latents_shape", ("1 " + " ".join(map(str, shape))).encode())
                txn.put(b"prompts_shape", b"1")
                txn.put(b"latents_0_data", row.tobytes())
                txn.put(b"prompts_0_data", b"a prompt")
            env.sync()
            env.close()
            dataset = CleanLatentLMDBDataset(directory)
            item = dataset[0]
            self.assertEqual(tuple(item["clean_latent"].shape), (21, 16, 60, 104))
            self.assertTrue(torch.equal(item["clean_latent"], torch.full_like(item["clean_latent"], 3.0)))
            self.assertEqual(item["prompts"], "a prompt")
            dataset.env.close()

    def test_reference_math_is_deterministic(self):
        import torch

        pool = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
            dtype=torch.float32,
        )
        torch.manual_seed(0)
        sigma_a = median_bandwidth(pool, max_subsample=4)
        torch.manual_seed(0)
        sigma_b = median_bandwidth(pool, max_subsample=4)
        self.assertEqual(sigma_a, sigma_b)
        bundle_a = build_nystrom_reference(
            pool,
            sigma_a,
            n_landmarks=2,
            fit_n=4,
            kmeans_iterations=2,
            krr_n=4,
            seed=0,
            device="cpu",
        )
        bundle_b = build_nystrom_reference(
            pool,
            sigma_a,
            n_landmarks=2,
            fit_n=4,
            kmeans_iterations=2,
            krr_n=4,
            seed=0,
            device="cpu",
        )
        self.assertTrue(torch.equal(bundle_a["Z"], bundle_b["Z"]))
        self.assertTrue(torch.equal(bundle_a["alpha"], bundle_b["alpha"]))
        self.assertEqual(bundle_a["k_rr"], bundle_b["k_rr"])

    def test_text_dataset_rejects_empty_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("valid\n\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                TextDataset(path)

    def test_inference_loader_accepts_fsdp_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "step": 20,
                    "generator": {
                        "model._fsdp_wrapped_module.test.weight": torch.ones(2)
                    },
                },
                checkpoint,
            )
            state = load_generator_state(str(checkpoint), use_ema=False)
            self.assertEqual(list(state), ["model.test.weight"])
            self.assertTrue(torch.equal(state["model.test.weight"], torch.ones(2)))

    def test_portable_checkpoint_strips_training_state(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "training.pt"
            packaged = Path(directory) / "generator.pt"
            torch.save(
                {
                    "step": 20,
                    "generator": {
                        "model._fsdp_wrapped_module.test.weight": torch.arange(4),
                    },
                    "trainer_variant": "internal_name",
                    "rdm_state": {"private_path": "/" + "mnt/private"},
                },
                source,
            )
            digest = package_checkpoint(
                source=source,
                destination=packaged,
            )
            self.assertEqual(len(digest), 64)
            payload = torch.load(
                packaged, map_location="cpu", weights_only=True, mmap=True
            )
            self.assertEqual(list(payload), ["generator"])
            self.assertEqual(list(payload["generator"]), ["model.test.weight"])
            self.assertNotIn("internal_name", payload)
            state = load_generator_state(str(packaged), use_ema=False)
            self.assertTrue(
                torch.equal(state["model.test.weight"], torch.arange(4))
            )


class EvaluationContractTest(unittest.TestCase):
    def test_prompt_deduplication_preserves_first_occurrence(self):
        rows, indices = unique_rows(["a", "b", "a", "c", "b"])
        self.assertEqual(rows, ["a", "b", "c"])
        self.assertEqual(indices, [0, 1, 3])

    def test_vbench_official_aggregation_extrema(self):
        minima = {name: limits[0] for name, limits in NORMALIZATION.items()}
        maxima = {name: limits[1] for name, limits in NORMALIZATION.items()}
        low = aggregate(minima)
        high = aggregate(maxima)
        self.assertAlmostEqual(low["total"], 0.0)
        self.assertAlmostEqual(high["quality"], 1.0)
        self.assertAlmostEqual(high["semantic"], 1.0)
        self.assertAlmostEqual(high["total"], 1.0)

    def test_partial_vbench_has_no_composite(self):
        summary = aggregate({"dynamic_degree": 0.5})
        self.assertIsNone(summary["quality"])
        self.assertIsNone(summary["semantic"])
        self.assertIsNone(summary["total"])
        self.assertEqual(summary["dynamic_degree"], 0.5)


if __name__ == "__main__":
    unittest.main()
