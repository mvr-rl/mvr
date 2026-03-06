# MVR: Multi-view Video Reward Shaping for Reinforcement Learning

Official implementation of **MVR**, accepted at **ICLR 2026**.

- Project page: `https://mvr-rl.github.io/`
- Paper (PDF): `https://mvr-rl.github.io/assets/MVR_ICLR2026.pdf`
- OpenReview: `https://openreview.net/forum?id=7lw6s9ELfr`

MVR learns state relevance from multi-view videos and turns it into reward shaping that helps early exploration while fading as the policy improves. This repository contains the JAX training pipeline, VLM-based reward shaping components, and scripts for MetaWorld and HumanoidBench experiments.

## Highlights

- Vision-language reward shaping with ViCLIP
- TQC-based reinforcement learning in JAX
- Multi-view video sampling and reward relabeling
- MetaWorld and HumanoidBench experiment scripts

## Installation

### Prerequisites

1. Python `3.11`
2. `uv` for environment management: `https://github.com/astral-sh/uv`
3. NVIDIA GPU with CUDA-compatible drivers for accelerated training
4. A C/C++ toolchain for packages that build native extensions

### Environment setup

From the repository root:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen --no-install-project
```

`--no-install-project` avoids installing this repository as an editable package, which prevents a module-name collision with ViCLIP's `utils` package during checkpoint loading.

### Verify JAX GPU setup

```bash
python -c "import jax; print('JAX version:', jax.__version__); print('JAX devices:', jax.devices()); print('JAX backend:', jax.default_backend())"
```

Expected output should show a CUDA device and `gpu` backend.

## Download pretrained ViCLIP weights

```bash
python download_viclip.py
```

The default checkpoint path is `ckpts/ViCLIP/ViCLIP-L_InternVid-FLT-10M.pth`.

## Running experiments

### MetaWorld MVR

```bash
bash scripts/mvr/run_mvr-metaworld.sh
```

### HumanoidBench MVR

```bash
bash scripts/mvr/run_mvr.sh
```

### TQC baselines

```bash
bash scripts/tqc/run_tqc_metaworld.sh
bash scripts/tqc/run_tqc.sh
```

Outputs are written under `outputs/<algo>/<timestamp>/<run_name>/<env_name>/`.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{luo2026mvr,
  title        = {MVR: Multi-view Video Reward Shaping for Reinforcement Learning},
  author       = {Luo, Lirui and Zhang, Guoxi and Xu, Hongming and Yang, Yaodong and Fang, Cong and Li, Qing},
  booktitle    = {International Conference on Learning Representations},
  year         = {2026}
}
```
