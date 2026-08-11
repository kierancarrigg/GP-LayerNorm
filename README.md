# GP-LayerNorm: Genetic Programming for LayerNorm Replacement

**Evolving Custom Normalization Functions for Vision Transformers**

This repository is the official implementation of *Evolving Layer-Specific Scalar Functions for Hardware-Aware Transformer Adaptation* ([arXiv:2605.14047](https://arxiv.org/abs/2605.14047)).
This repository implements a genetic programming (GP) approach to evolve custom replacement functions for LayerNorm in vision transformers. Building upon the DyT (Dynamic Tanh) work from "Transformers without Normalization" (CVPR 2025), we explore whether GP can discover more effective normalization alternatives beyond simple parametric forms.

## Overview

LayerNorm is a critical component in transformers, but recent work (DyT) has shown that simpler operations like `tanh(α·x)` can achieve comparable performance. This project investigates:

1. **Can GP evolve better normalization replacements than hand-designed functions?**
2. **What mathematical structures emerge from evolution on real LayerNorm I/O data?**
3. **How do evolved functions compare to DyT in terms of accuracy, efficiency, and generalization?**

## Key Contributions

- **GP Evolution Pipeline**: Extract LN I/O mappings → Evolve functions → Generate PyTorch layers
- **Finetuning Framework**: Efficient finetuning strategies for both GP-evolved and DyT models


GP evolution is powered by [Kozax](https://github.com/sdevries0/Kozax), an external JAX-based genetic programming library. This project uses Kozax's FLOP-based complexity feature, which is not yet in a PyPI release — see [Installation](#installation) for the pinned git install.

## Project Structure

```
GP-LayerNorm/
├── gp/                     # GP-based approach
│   ├── evolution/          # Evolution pipeline (main.py, fitness.py, operators.py, extract_mappings.py)
│   ├── layers/             # Evolved layer implementations (evolved_layers_seed_1.py – seed_5.py)
│   └── finetune/           # GP model finetuning (train.py)
├── dyt_finetune/           # DyT finetuning (train.py)
├── baselines/              # Baseline implementations (dyt.py — MIT licensed, adapted from DyT)
├── utils/                  # Shared utilities (datasets.py, training.py, helpers.py)
├── scripts/                # SLURM job script templates
└── requirements.txt        # pip dependencies (kozax installed separately from git, see Installation)
```

## Installation

### Prerequisites
- Python 3.12
- CUDA 12.4 (for GPU support)
- Conda (recommended)

### Setup

```bash
# Create conda environment
conda create -n gp-layernorm python=3.12
conda activate gp-layernorm

# Install PyTorch
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install kozax from a pinned commit — its FLOP-based complexity feature isn't
# on PyPI yet, and kozax doesn't track packaging metadata in git, so we supply
# a minimal pyproject.toml ourselves (see note below)
git clone -q https://github.com/sdevries0/Kozax.git /tmp/kozax && cd /tmp/kozax && git checkout -q 8fa0187
printf '[build-system]\nrequires = ["setuptools>=61.0"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "kozax"\nversion = "0.1.4+g8fa0187"\n\n[tool.setuptools.packages.find]\ninclude = ["kozax*"]\n' > pyproject.toml
pip install --no-deps . && cd - && rm -rf /tmp/kozax

# Install remaining dependencies
pip install -r requirements.txt
```

> **Why not a plain `pip install kozax`?** This project needs Kozax's FLOP-based
> complexity objective, which isn't in any PyPI release yet (latest is 0.1.4, which
> predates the upstream commit that added it) — so we pin the exact commit
> (`8fa0187`) this project's results were generated against. That commit has no
> `pyproject.toml`: kozax's own `.gitignore` excludes it from version control, so
> the snippet above supplies a minimal one ourselves — it's a scratch clone, not
> a fork or a modification to kozax's repository.

## Usage

> For SLURM cluster users, see `scripts/` for ready-to-adapt job scripts.

<details>
<summary><strong>1. Extract LayerNorm I/O Mappings</strong></summary>

Extract input-output pairs from a pretrained ViT-B using ImageNet validation data:

```bash
python gp/evolution/extract_mappings.py \
    --model_name vit_base_patch16_224 \
    --imagenet_root /path/to/imagenet/val \
    --batch_size 64 \
    --points_per_forward 50000 \
    --output_file ln_mappings.npz
```

</details>

<details>
<summary><strong>2. Evolve Normalization Functions</strong></summary>

Run GP evolution to discover replacement functions for all 25 LN positions in ViT-B:

```bash
python gp/evolution/main.py \
    --data_file ln_mappings.npz \
    --output_csv gp_results.csv \
    --num_generations 50 \
    --population_size 500 \
    --num_populations 1 \
    --max_nodes 20 \
    --max_init_depth 4 \
    --tournament_size 7 \
    --penalty_weight 0.005 \
    --complexity_objective true \
    --constant_optimization true \
    --constant_optimization_steps 10
```

Complexity is measured in **FLOPs per operator** (`tanh` 23, `sigmoid` 22, `+`/`*` 1, `neg` 0, `clip` 1),
so evolution trades accuracy against real hardware cost rather than raw expression size.
Each Pareto-front solution is written to the results CSV with its exact FLOP count.

`clip` is the one deliberate exception: its true cost is 0 FLOPs (a bounds check, no
arithmetic), and 0 is what every reported figure uses. It is charged 1 FLOP *during the
search only*, as a regulariser — at zero cost, GP stacks redundant clips for free.

</details>

<details>
<summary><strong>3. Generate PyTorch Layers</strong></summary>

Convert evolved GP expressions from the results CSV into PyTorch module files:

```bash
python gp/layers/generate_code.py \
    --input_csv gp_results.csv \
    --output_dir gp/layers/ \
    --strategy kneedle
```

This generates `gp/layers/evolved_layers_seed_1.py` through `evolved_layers_seed_5.py`.
The pre-generated files used in the paper are already included in the repo — you only need this step if you run new evolution.

`--strategy` controls how one solution is picked from each layer's Pareto front:

| Strategy | Behaviour |
|---|---|
| `kneedle` (default) | Knee point of the (FLOPs, MSE) front — the elbow beyond which extra FLOPs buy little accuracy |
| `min_mse` | Lowest MSE regardless of cost |

</details>

See [Reproducing Paper Results](#reproducing-paper-results) for finetuning commands with the exact hyperparameters used in the paper.

## Reproducing Paper Results

All experiments use `vit_base_patch16_224` pretrained on ImageNet. Replace `/path/to/imagenet` with your ImageNet root. GP variants are run independently for seeds 1–5.

> All commands below match the exact hyperparameters of the runs behind the results table.

### GP Variants

<details>
<summary><strong>GP-A</strong> — affine-only (frozen backbone, only evolved layer params trained)</summary>

```bash
python gp/finetune/train.py \
    --gp_seed 1 \
    --data_path /path/to/imagenet \
    --lr 1e-3 --lr_scheduler none \
    --weight_decay 0.0 --drop_path 0.0 \
    --output_dir ./checkpoints/gp_a_seed1
```

</details>

<details>
<summary><strong>GP-F</strong> — full model finetuning</summary>

```bash
python gp/finetune/train.py \
    --gp_seed 1 \
    --data_path /path/to/imagenet \
    --train_mode full \
    --lr 1e-5 --lr_scheduler cosine \
    --weight_decay 0.0 --drop_path 0.0 \
    --output_dir ./checkpoints/gp_f_seed1
```

</details>

<details>
<summary><strong>GP-D</strong> — full model + logit distillation from pretrained LN teacher</summary>

```bash
python gp/finetune/train.py \
    --gp_seed 1 \
    --data_path /path/to/imagenet \
    --train_mode full --distill_logit true \
    --lr 1e-5 --lr_scheduler cosine \
    --weight_decay 0.05 --drop_path 0.2 \
    --output_dir ./checkpoints/gp_d_seed1
```

</details>

### DyT Variants

<details>
<summary><strong>DyT-A</strong> — affine-only</summary>

```bash
python dyt_finetune/train.py \
    --data_path /path/to/imagenet \
    --train_mode affine \
    --lr 8e-3 --lr_scheduler cosine \
    --output_dir ./checkpoints/dyt_a
```

</details>

<details>
<summary><strong>DyT-F</strong> — full model with per-group learning rates</summary>

```bash
python dyt_finetune/train.py \
    --data_path /path/to/imagenet \
    --train_mode full \
    --lr_backbone 2e-5 --lr_affine 1e-4 --lr_alpha 5e-5 \
    --lr_scheduler cosine \
    --output_dir ./checkpoints/dyt_f
```

</details>

<details>
<summary><strong>DyT-D</strong> — full model + logit distillation</summary>

```bash
python dyt_finetune/train.py \
    --data_path /path/to/imagenet \
    --train_mode full --distill_logit true \
    --lr_backbone 3e-5 --lr_affine 1e-4 --lr_alpha 5e-5 \
    --lr_scheduler cosine \
    --output_dir ./checkpoints/dyt_d
```

</details>

### LN Baseline

<details>
<summary><strong>LN</strong> — full finetuning without normalization replacement</summary>

```bash
python dyt_finetune/train.py \
    --data_path /path/to/imagenet \
    --use_dyt false --train_mode full \
    --lr 1e-6 --lr_scheduler cosine \
    --weight_decay 0.05 --drop_path 0.2 \
    --output_dir ./checkpoints/ln
```

</details>

## Results

Classification performance of the evolved symbolic normalizations (GP) against standard LayerNorm and
Dynamic Tanh (DyT) baselines on the ImageNet-1K validation set, using a pretrained ViT-B architecture.
All fine-tuned variants are trained for 20 epochs and evaluated across 5 independent runs (mean ± std).
Bold marks the better method within each fine-tuning regime.

| Method | Top-1 Acc (%) | Top-5 Acc (%) |
|--------|--------------|---------------|
| ***Literature & reference baselines*** | | |
| Pre-trained ViT-B (no fine-tuning) | 80.99 | 95.73 |
| Original DyT (trained from scratch) † | 82.5 | — |
| Standard ViT-B (LN fine-tuning) | 84.99 ± 0.02 | 97.43 ± 0.02 |
| ***Affine-only fine-tuning*** | | |
| DyT-A | **83.19 ± 0.03** | **96.74 ± 0.02** |
| GP-A (Ours) | 82.48 ± 0.20 | 96.43 ± 0.06 |
| ***Full fine-tuning*** | | |
| DyT-F | 82.81 ± 0.02 | 96.60 ± 0.02 |
| GP-F (Ours) | **84.07 ± 0.06** | **97.10 ± 0.03** |
| ***Knowledge distillation*** | | |
| DyT-D | 83.36 ± 0.04 | 96.79 ± 0.05 |
| GP-D (Ours) | **84.32 ± 0.01** | **97.15 ± 0.01** |

† Reported supervised classification accuracy from the original DyT paper (Zhu et al., CVPR 2025).

## Citation

If you use this work in your research, please cite:

```bibtex
@article{carrigg2026gplayernorm,
  title={Evolving Layer-Specific Scalar Functions for Hardware-Aware Transformer Adaptation},
  author={Carrigg, Kieran and de Vries, Sigur and Sadough, Amirhossein and van Gerven, Marcel},
  journal={arXiv preprint arXiv:2605.14047},
  year={2026}
}
```

Please also cite the original DyT work this builds upon:

```bibtex
@inproceedings{zhu2025dyt,
  title={Transformers without Normalization},
  author={Zhu, Jiachen and Chen, Xinlei and He, Kaiming and LeCun, Yann and Liu, Zhuang},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgments

This project builds upon the excellent work of:
- **DyT** ([Transformers without Normalization](https://github.com/jiachenzhu/DyT)) by Jiachen Zhu et al.
- **Kozax** ([sdevries0/Kozax](https://github.com/sdevries0/Kozax)) — the JAX-based genetic programming library powering GP evolution
- **timm** library by Ross Wightman
- JAX ecosystem

See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed acknowledgments and license information.

## License

Original contributions in this project are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Copyright 2026 Kieran Carrigg).

Adapted DyT components (`baselines/dyt.py`, `utils/helpers.py`, `utils/training.py`, `utils/datasets.py`) retain their original MIT License from Meta Platforms, Inc. and affiliates.

See [LICENSE](LICENSE) for the full dual-license details.
