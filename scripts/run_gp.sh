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

# --- Architecture ---
# --gp_arch selects which evolved layers to load from gp/layers/ and must match --model.
# --batch_size is per GPU: ViT-B ran on 1 GPU at 512, ViT-L on 2 GPUs at 256, so both
# give a global batch of 512. The GP-A learning rate and schedule differ between scales.
MODEL=vit_base_patch16_224
GP_ARCH=vit_b
NUM_GPUS=1
BATCH_SIZE=512
LR=1e-3
LR_SCHEDULER=none
# For ViT-L, set --gres=gpu:2 above and use:
#   MODEL=vit_large_patch16_224
#   GP_ARCH=vit_l
#   NUM_GPUS=2
#   BATCH_SIZE=256
#   LR=2e-3
#   LR_SCHEDULER=cosine

torchrun --nproc_per_node=$NUM_GPUS gp/finetune/train.py \
    --gp_seed $GP_SEED \
    --gp_arch $GP_ARCH \
    --epochs 20 \
    --data_path $IMAGENET_PATH \
    --model $MODEL \
    --lr $LR \
    --lr_scheduler $LR_SCHEDULER \
    --train_mode affine \
    --weight_decay 0.0 \
    --batch_size $BATCH_SIZE \
    --num_workers 8 \
    --drop_path 0.0 \
    --use_amp true \
    --enable_wandb false \
    --output_dir $OUTPUT_DIR/${SLURM_JOB_ID}
