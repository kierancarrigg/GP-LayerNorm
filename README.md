# GP-LayerNorm: Genetic Programming for LayerNorm Replacement

**Evolving Custom Normalization Functions for Vision Transformers**

This repository is the official implementation of *Evolving Layer-Specific Scalar Functions for Hardware-Aware Transformer Adaptation* (arXiv link to be added).
This repository implements a genetic programming (GP) approach to evolve custom replacement functions for LayerNorm in vision transformers. Building upon the DyT (Dynamic Tanh) work from "Transformers without Normalization" (CVPR 2025), we explore whether GP can discover more effective normalization alternatives beyond simple parametric forms.

## Overview

LayerNorm is a critical component in transformers, but recent work (DyT) has shown that simpler operations like `tanh(α·x)` can achieve comparable performance. This project investigates:

1. **Can GP evolve better normalization replacements than hand-designed functions?**
2. **What mathematical structures emerge from evolution on real LayerNorm I/O data?**
3. **How do evolved functions compare to DyT in terms of accuracy, efficiency, and generalization?**

## Key Contributions

- **GP Evolution Pipeline**: Extract LN I/O mappings → Evolve functions → Generate PyTorch layers
- **Finetuning Framework**: Efficient finetuning strategies for both GP-evolved and DyT models


GP evolution is powered by [Kozax](https://github.com/kc-ml2/kozax), an external JAX-based genetic programming library (`pip install kozax`).

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
└── requirements.txt        # pip dependencies (includes kozax>=0.0.13)
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

# Install kozax separately first (avoids a sympy version conflict with torch's PyPI metadata)
pip install kozax==0.0.13 --no-deps

# Install remaining dependencies
pip install -r requirements.txt
```

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
    --constant_optimization_method gradient \
    --constant_optimization_steps 10
```

</details>

<details>
<summary><strong>3. Generate PyTorch Layers</strong></summary>

Convert evolved GP expressions from the results CSV into PyTorch module files:

```bash
python gp/layers/generate_code.py \
    --input_csv gp_results.csv \
    --output_dir gp/layers/
```

This generates `gp/layers/evolved_layers_seed_1.py` through `evolved_layers_seed_5.py`.
The pre-generated files used in the paper are already included in the repo — you only need this step if you run new evolution.

</details>

See [Reproducing Paper Results](#reproducing-paper-results) for finetuning commands with the exact hyperparameters used in the paper.

## Reproducing Paper Results

All experiments use `vit_base_patch16_224` pretrained on ImageNet. Replace `/path/to/imagenet` with your ImageNet root. GP variants are run independently for seeds 1–5.

### GP Variants

<details>
<summary><strong>GP-A</strong> — affine-only (frozen backbone, only evolved layer params trained)</summary>

```bash
python gp/finetune/train.py \
    --gp_seed 1 \
    --data_path /path/to/imagenet \
    --lr 2e-3 --weight_decay 0.0 --drop_path 0.0 \
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
    --lr 1e-5 --weight_decay 0.0 \
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
    --lr 1e-5 --weight_decay 0.0 \
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
    --lr 8e-3 \
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
    --lr 1e-6 --drop_path 0.1 \
    --output_dir ./checkpoints/ln
```

</details>

## Results

Results on the ImageNet-1K validation set, averaged across 5 independent runs (ViT-B/16, 20 epochs fine-tuning).

| Method | Top-1 Acc (%) | Top-5 Acc (%) |
|--------|--------------|---------------|
| LN (Full fine-tuning) | 84.94 ± 0.01 | 97.43 ± 0.01 |
| | | |
| DyT-A | 82.99 ± 0.07 | 96.65 ± 0.02 |
| GP-A (Ours) | 82.78 ± 0.08 | 96.58 ± 0.04 |
| | | |
| DyT-F | 82.12 ± 0.05 | 96.32 ± 0.03 |
| GP-F (Ours) | **83.70 ± 0.04** | **96.99 ± 0.02** |
| | | |
| DyT-D | 82.66 ± 0.09 | 96.56 ± 0.03 |
| GP-D (Ours) | **84.25 ± 0.02** | **97.18 ± 0.02** |

## Citation

If you use this work in your research, please cite:

```bibtex
@article{carrigg2026gplayernorm,
  title={Evolving Layer-Specific Scalar Functions for Hardware-Aware Transformer Adaptation},
  author={Carrigg, Kieran},
  journal={arXiv preprint arXiv:XXXX.XXXXX},  % TODO: fill in arXiv ID after submission
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
- **timm** library by Ross Wightman
- JAX ecosystem

See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed acknowledgments and license information.

## License

Original contributions in this project are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Copyright 2026 Kieran Carrigg).

Adapted DyT components (`baselines/dyt.py`, `utils/helpers.py`, `utils/training.py`, `utils/datasets.py`) retain their original MIT License from Meta Platforms, Inc. and affiliates.

See [LICENSE](LICENSE) for the full dual-license details.
