#!/bin/bash
#SBATCH --job-name=eval_pseudo
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/data/cashel-data/marvel-maritime-images/eval_pseudo_%j.out

echo "Changing directory to shared path..."
cd /data/cashel-data/Co-DETR

echo "Activating codetr conda environment..."
# Properly initialize conda for non-interactive slurm scripts
eval "$(conda shell.bash hook)"
conda activate codetr

echo "Setting PYTHONPATH..."
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Increasing NCCL Timeout to 4 hours (14400 seconds) to prevent watchdog timeout..."
export NCCL_TIMEOUT=14400

# Create tmpdir if it doesn't exist
TMP_DIR="/data/cashel-data/marvel-maritime-images/temp_inference_parts"
mkdir -p $TMP_DIR

echo "Starting distributed inference via tools/dist_test.sh..."
# Note: Since the work_dir was excluded in rsync, we read the checkpoint from the original path
CHECKPOINT_PATH="checkpoints/best_bbox_mAP_epoch_4.pth"

bash tools/dist_test.sh \
    projects/configs/co_dino_vit/marvel_maritime_pseudo.py \
    $CHECKPOINT_PATH \
    8 \
    --format-only \
    --tmpdir $TMP_DIR \
    --eval-options "jsonfile_prefix=/data/cashel-data/marvel-maritime-images/pseudo_best_bbox_mAP_epoch_4"

echo "Inference finished."
