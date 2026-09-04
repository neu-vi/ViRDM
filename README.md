<h1 align="center">ViRDM</h1>
<h3 align="center">Taming Representation Distribution Matching for<br>Few-Step Causal Video Generation</h3>

<p align="center">
  <a href="https://cr8br0ze.github.io/">Zichong Meng</a>,
  <a href="https://chongjiange.github.io/">Chongjian Ge</a>,
  <a href="https://paulchhuang.github.io/">Chun-Hao P. Huang</a>,
  <a href="https://yzhou359.github.io/">Yang Zhou</a><sup>&dagger;</sup>,
  <a href="https://jianghz.me/">Huaizu Jiang</a><sup>&dagger;</sup>
</p>
<p align="center">Northeastern University · Adobe Research</p>
<p align="center"><sub><sup>&dagger;</sup> Equal advising</sub></p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.28923"><img src="https://img.shields.io/badge/arXiv-2609.28923-A42C25?style=flat&amp;logo=arxiv&amp;logoColor=white" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2609.28923"><img src="https://img.shields.io/badge/Paper-PDF-yellow?style=flat&amp;logo=arxiv&amp;logoColor=white" alt="Paper PDF"></a>
  <a href="https://neu-vi.github.io/ViRDM/"><img src="https://img.shields.io/badge/Project-Page-orange?style=flat&amp;logo=googlechrome&amp;logoColor=white" alt="Project page"></a>
  <a href="https://github.com/neu-vi/ViRDM"><img src="https://img.shields.io/badge/GitHub-Code-black?style=flat&amp;logo=github&amp;logoColor=white" alt="GitHub code"></a>
  <a href="https://huggingface.co/cr8br0ze/ViRDM"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=white" alt="Hugging Face models"></a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=neu-vi.ViRDM&amp;left_color=gray&amp;right_color=blue" alt="Visitors">
</p>

---

ViRDM enables **teacher- and critic-free post-training** for few-step causal
video generation by matching generated videos directly to a fixed video-text
representation distribution. With only **20 generator updates**, it reaches
**84.87 VBench Total** while reducing peak memory from 77.1 to 48.3 GB per GPU
and post-training time from 22 to 2 hours on eight A100 GPUs.

## Supported recipes

The reported training recipes use eight GPUs, per-GPU batch size 1, gradient
accumulation 8 (global generated population 64), learning rate `2e-6`, and 20
generator updates. A one-GPU launcher preserves the same population with
gradient accumulation 64. Only step 20 is saved.

| Training name | Schedule | Inference name |
|---|---|---|
| `chunk4` | causal chunkwise 4-step | `causal4` |
| `chunk4+2` | causal chunkwise 2-step with a 4-step first chunk | `causal2` |
| `chunk4+1` | causal chunkwise 1-step with a 4-step first chunk | `causal1` |
| `bid4` | bidirectional 4-step | `bid4` |

Each training name has `nodynamic` and `dynamic` variants. The latter adds the
paper's dynamic regularizer with weight `5e-4`; it does not change the ViRDM
distribution objective.

## 1. Environment

The supported installation uses Conda on Python 3.10, CUDA
12.4, PyTorch 2.6, and NVIDIA GPUs with BF16 support. The reference training
configuration uses 8 x 80GB A100 GPUs.

~~~bash
git clone https://github.com/neu-vi/ViRDM.git
cd ViRDM
conda env create -f environment.yml
conda activate virdm
bash scripts/setup_env.sh
bash scripts/test_release.sh
~~~

## 2. Artifacts

### Inference artifacts

This downloads the official Wan2.1-T2V-1.3B runtime, the public Causal-ODE
initialization, and the released causal four-step ViRDM generators (with and
without dynamics regularization).

```bash
python scripts/download_artifacts.py --scope inference
```

Inference artifacts layout:

```text
artifacts/
├── checkpoints/chunkwise/causal_ode.pt
├── checkpoints/virdm/
│   ├── virdm_causal4_dynamic_step20.pt
│   └── virdm_causal4_nodynamic_step20.pt
└── wan/Wan2.1-T2V-1.3B/
    ├── config.json
    ├── diffusion_pytorch_model.safetensors
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── Wan2.1_VAE.pth
    └── google/umt5-xxl/
```

### Training artifacts

Training additionally needs V-JEPA 2, TAEW2.1, the 6,505-row prompt table, the frozen joint reference, its aligned SigLIP2 table, and the frozen dense-flow checkpoint for dynamic regularization.

```bash
python scripts/download_artifacts.py --scope training --dynamic
```

<details>
<summary>(Optional) Rebuild the prompt-only LMDB</summary>

The prompt-only LMDB can alternatively be reconstructed without changing the
official dataset. Download `zhuhz22/Causal-Forcing-data`, then point the script
at either its 15 original chunkwise shards or its merged `clean_data` folder:

```bash
hf download zhuhz22/Causal-Forcing-data --local-dir causal-forcing-data
python scripts/prepare_prompt_lmdb.py \
  --source causal-forcing-data \
  --output artifacts/prompt_data
```

The script copies captions only, preserves the official shard order. ViRDM never reads training
latents or videos during its generator update.

</details>

<details>
<summary>Technical details: Frozen-reference distribution</summary>

The released reference was computed once from all 6,505 clean videos.

Each 21-frame latent is decoded with Wan's VAE into an
81-frame, 480×832 video, padded to 82 frames by repeating the final frame, and
encoded by the frozen V-JEPA 2.1 ViT-L/16 EMA encoder. Its final-layer-normalized
tokens (41×30×52 per video) are averaged into one 1,024-dimensional visual
feature. The aligned prompt is encoded by SigLIP2 into a unit-normalized
1,152-dimensional text feature. The joint row is
`[video | 23.77343595 × text]`.

The Gaussian bandwidth is the visual median distance at scale 1
(`sigma=26.08997917`); the text multiplier is that bandwidth divided by the
text median distance (`1.09744251`). The published file stores the 4,096
Nyström landmarks, attraction coefficients, fixed real-real kernel term, and
the full preprocessing contract.

</details>

<details>
<summary>(Optional) Rebuild the frozen reference</summary>
The reference can be recomputed from the original clean-latent data; no released
feature table is needed. First download the three frozen components used only by
this offline job and the official data:

```bash
python scripts/download_artifacts.py --scope reference
hf download zhuhz22/Causal-Forcing-data --local-dir causal-forcing-data
```

Then run the builder on one or eight GPUs. It accepts either the 15 downloaded
shards or a merged `clean_data` LMDB and preserves their canonical row order:

```bash
torchrun --standalone --nproc_per_node=8 scripts/build_reference.py \
  --source causal-forcing-data \
  --output-dir artifacts/rebuilt_reference
```

For every row, the builder selects the final clean latent, decodes its 21 latent
frames to 81 RGB frames with Wan's VAE, and extracts the global mean of all
final-layer V-JEPA tokens. Independently, it re-tokenizes and re-encodes every
aligned prompt with the frozen SigLIP2 text tower and L2-normalizes each text
row. It resets seed 0 for both median estimates, computes
`beta = sigma_video / s_text`, forms `[video | beta * text]`, and fits the 4,096
landmark Nyström reference. The output includes both source feature tables,
an aligned prompt-only LMDB, receipts, and
`config_virdm_rebuilt_reference.yaml`, which can be passed as
`BASE_CONFIG` to a training launcher. Add `--skip-existing` when restarting a
partially completed extraction.

</details>

### Full training layout

```text
artifacts/
├── checkpoints/chunkwise/causal_ode.pt
├── prompt_data/data.mdb
├── references/reference_M4096.pt
├── references/siglip2_text_fp32.npy
├── vjepa/vjepa2_1_vitl_dist_vitG_384.pt
├── taew/taew2_1.pth
├── flow/dense_flow.pth                  # dynamic only
└── wan/Wan2.1-T2V-1.3B/...
```

## 3. Quick inference

Use the released model with dynamics regularization:

```bash
# Replace prompts/example.txt with prompts/demos.txt for more prompts.
bash scripts/infer.sh \
  --recipe causal4 \
  --checkpoint_path artifacts/checkpoints/virdm/virdm_causal4_dynamic_step20.pt \
  --prompt_path prompts/example.txt \
  --output_folder outputs/virdm_causal4_dynamic \
  --seed $RANDOM \
  --gpu_id 0
```

Substitute
`virdm_causal4_nodynamic_step20.pt` for the model without dynamics
regularization.

## 4. Training
```bash
wandb login
# export WANDB_MODE=offline   # no network upload
# export WANDB_MODE=disabled  # disable W&B entirely

bash scripts/train_1x8.sh chunk4 nodynamic
bash scripts/train_1x8.sh chunk4 dynamic
bash scripts/train_1x8.sh chunk4+2 nodynamic
bash scripts/train_1x8.sh chunk4+2 dynamic
bash scripts/train_1x8.sh chunk4+1 nodynamic
bash scripts/train_1x8.sh chunk4+1 dynamic
bash scripts/train_1x8.sh bid4 nodynamic
bash scripts/train_1x8.sh bid4 dynamic
```

The same objective and global population can run on one 80GB GPU:

```bash
bash scripts/train_1gpu.sh chunk4 nodynamic
bash scripts/train_1gpu.sh chunk4 dynamic
```

All four training names and both dynamics modes are accepted by
`scripts/train_1gpu.sh`.

Outputs are written to `runs/<run-name>/`. The final generator checkpoint is:

```text
runs/<run-name>/checkpoint_model_000020/model.pt
```

## 5. Inference
```bash
# Chunkwise 4-step model
bash scripts/infer.sh --recipe causal4 \
  --checkpoint_path artifacts/checkpoints/virdm/virdm_causal4_dynamic_step20.pt \
  --prompt_path prompts/example.txt --output_folder outputs/causal4

# Bidirectional 4-step model
bash scripts/infer.sh --recipe bid4 \
  --checkpoint_path runs/virdm_bid4_nodynamic_bs64_lr2e6_step20/checkpoint_model_000020/model.pt \
  --prompt_path prompts/example.txt --output_folder outputs/bid4
```

Pass `--gpus 8` to shard prompts across eight GPUs, `--seed N` to change the
base seed, and `--num_samples_per_prompt N` for repeated samples. One non-empty
prompt is expected per line. Existing output videos are skipped, so an
interrupted prompt sweep can be restarted with the same command.

## 6. VBench evaluation

<details>
<summary>(Optional) VBench evaluation instructions</summary>

VBench uses a separate Conda environment with Python 3.10, CUDA toolkit 12.1,
and PyTorch 2.4.1. Do not install its dependencies into the training environment.
The helper pins VBench and Detectron2 source revisions and compiles Detectron2
with the evaluator environment's toolkit and compiler.

~~~bash
conda env create -f environment-vbench.yml
conda activate virdm-vbench
bash scripts/setup_vbench_env.sh
~~~

VBench is checked out under external/VBench. Switch back to virdm to generate
videos, then activate virdm-vbench to score them.

Generate the official 944 unique prompts in the ViRDM environment:

```bash
conda activate virdm
bash scripts/generate_vbench.sh \
  --recipe causal4 \
  --checkpoint artifacts/checkpoints/virdm/virdm_causal4_dynamic_step20.pt \
  --vbench-root external/VBench \
  --extended-prompts prompts/vbench/all_dimension_extended.txt \
  --output outputs/vbench/causal4 \
  --gpus 8 \
  --samples-per-prompt 5
```

The repository includes the exact 946-row Self-Forcing VBench prompt pair:
`prompts/vbench/all_dimension.txt` contains the original prompts used for file
names and `all_dimension_extended.txt` contains the line-aligned generation
text used by the matched evaluation protocol. The generation helper derives
file names from VBench metadata, aligns the extended text by row, and resolves
the two duplicate rows into 944 unique prompt/video names.

`--samples-per-prompt 5` is the standard VBench protocol (4,720 videos).

Score all 16 official dimensions in the VBench environment:

```bash
conda activate virdm-vbench
bash scripts/eval_vbench.sh \
  --videos outputs/vbench/causal4/videos \
  --vbench-root external/VBench \
  --name virdm_causal4 \
  --output eval/virdm_causal4 \
  --local-models \
  --gpus 8
```

To evaluate a subset, put `--dimensions` last:

```bash
bash scripts/eval_vbench.sh \
  --videos VIDEO_DIR --vbench-root external/VBench --name dynamic_only \
  --local-models --gpus 8 \
  --dimensions dynamic_degree
```

</details>

## Acknowledgements

We thank the authors of
[Wan2.1](https://github.com/Wan-Video/Wan2.1),
[Causal Forcing](https://github.com/thu-ml/Causal-Forcing),
[Self-Forcing](https://github.com/guandeh17/Self-Forcing), and
[One-Forcing](https://github.com/Aurora-edu/One-Forcing) for making their work
publicly available. We also thank the authors of
[V-JEPA 2](https://github.com/facebookresearch/vjepa2),
[TAEW2.1](https://github.com/madebyollin/taehv),
[RAFT](https://github.com/princeton-vl/RAFT),
[SigLIP2](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-256), and
[VBench](https://github.com/Vchitect/VBench). See
[`THIRD_PARTY.md`](THIRD_PARTY.md) for source and license details. Model weights
and datasets remain subject to their respective upstream terms.

## Citation

```bibtex
@article{meng2026virdm,
  title   = {ViRDM: Taming Representation Distribution Matching for
             Few-Step Causal Video Generation},
  author  = {Meng, Zichong and Ge, Chongjian and Huang, Chun-Hao P. and Zhou, Yang and Jiang, Huaizu},
  journal = {arXiv preprint arXiv:2609.28923},
  year    = {2026}
}
```
