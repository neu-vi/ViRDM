# Third-party components

ViRDM contains adapted source from the following projects. Their original
license texts are retained in the repository.

| Component | Upstream | Use | License |
|---|---|---|---|
| Wan2.1 | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | generator, text encoder, VAE | Apache-2.0 |
| FlashAttention 2 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) `2.8.3.post1` | causal generator attention backend used for reported training runtime | BSD-3-Clause |
| Self-Forcing | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) at `33593df3e81fa3ec10239271dd2c100facac6de1` | causal generator architecture, stochastic-exit rollout, and extended VBench generation prompts | Apache-2.0 |
| Causal Forcing | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) | chunkwise/full-video recipes, public Causal-ODE initialization, and demo prompt suite | Apache-2.0 |
| V-JEPA 2 | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) at `204698b45b3712590f06245fbfba32d3be539812` | frozen video representation encoder | MIT / Apache-2.0 |
| TAEHV / TAEW2.1 | [madebyollin/taehv](https://github.com/madebyollin/taehv) at `e743234f3217ab3d1570f65642ab06596d1bd7c5` | lightweight differentiable decoder | MIT |
| RAFT | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | optional frozen dense-flow regularizer | BSD-3-Clause |
| VBench | [Vchitect/VBench](https://github.com/Vchitect/VBench) | evaluation only; installed separately | project license |

The bundled, import-rewritten subsets live under `third_party/`. Model weights
are not committed and remain governed by their upstream licenses and model
terms. Exact source revisions, download locations, and file checksums are
recorded in [`artifacts/manifest.json`](artifacts/manifest.json).
