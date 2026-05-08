#!/bin/bash
#SBATCH --job-name=extract_mappings
#SBATCH -p <SLURM_PARTITION>
#SBATCH -c 10
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40gb
#SBATCH --output=logs/extract_mappings_%j.out
#SBATCH --error=logs/extract_mappings_%j.err

mkdir -p logs

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gp-layernorm

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- Configuration ---
IMAGENET_VAL=<IMAGENET_VAL_PATH>  # e.g. /data/imagenet/val

python gp/evolution/extract_mappings.py \
    --seed 42 \
    --imagenet_root $IMAGENET_VAL \
    --model_name vit_base_patch16_224 \
    --batch_size 64 \
    --points_per_forward 50000 \
    --output_file ln_mappings.npz
