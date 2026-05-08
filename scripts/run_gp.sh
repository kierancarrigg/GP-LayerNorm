#!/bin/bash
#SBATCH --job-name=gp_finetune
#SBATCH -p <SLURM_PARTITION>
#SBATCH -c 9
#SBATCH -t 6-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --output=logs/gp_finetune-%J.out
#SBATCH --error=logs/gp_finetune-%J.err

mkdir -p logs

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gp-layernorm

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export PYTHONPATH=$PYTHONPATH:.
export PYTHONDONTWRITEBYTECODE=1

# --- Configuration ---
IMAGENET_PATH=<IMAGENET_PATH>     # e.g. /data/imagenet
OUTPUT_DIR=<OUTPUT_DIR>           # e.g. /scratch/outputs
GP_SEED=${SLURM_ARRAY_TASK_ID:-1} # Seeds 1-5 correspond to the 5 evolved layer files

torchrun --nproc_per_node=1 gp/finetune/train.py \
    --gp_seed $GP_SEED \
    --epochs 20 \
    --data_path $IMAGENET_PATH \
    --model vit_base_patch16_224 \
    --lr 2e-3 \
    --train_mode affine \
    --weight_decay 0.0 \
    --batch_size 512 \
    --num_workers 8 \
    --drop_path 0.0 \
    --use_amp true \
    --enable_wandb false \
    --output_dir $OUTPUT_DIR/${SLURM_JOB_ID}
