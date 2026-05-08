#!/bin/bash
#SBATCH --job-name=dyt_finetune
#SBATCH -p <SLURM_PARTITION>
#SBATCH -c 9
#SBATCH -t 6-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --output=logs/dyt_finetune-%J.out
#SBATCH --error=logs/dyt_finetune-%J.err

mkdir -p logs

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gp-layernorm

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# --- Configuration ---
IMAGENET_PATH=<IMAGENET_PATH>     # e.g. /data/imagenet
OUTPUT_DIR=<OUTPUT_DIR>           # e.g. /scratch/outputs
SEED=${SLURM_ARRAY_TASK_ID:-0}

torchrun --nproc_per_node=1 dyt_finetune/train.py \
    --seed $SEED \
    --data_path $IMAGENET_PATH \
    --model vit_base_patch16_224 \
    --train_mode affine \
    --use_dyt true \
    --epochs 20 \
    --batch_size 512 \
    --num_workers 8 \
    --use_amp true \
    --lr 8e-3 \
    --weight_decay 0.0 \
    --drop_path 0.0 \
    --enable_wandb false \
    --output_dir $OUTPUT_DIR/${SLURM_JOB_ID}
