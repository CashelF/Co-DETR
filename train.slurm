#!/bin/bash

############################
# Slurm job configuration #
############################

#SBATCH --job-name=co_dino_train
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=1024G
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --mail-user=cash@modernintelligence.ai

#SBATCH --output=/data/codetr-output-dir/logs/%x_%j.out
#SBATCH --error=/data/codetr-output-dir/logs/%x_%j.err

#####################
# Environment setup #
#####################

# Fail fast
set -euo pipefail

# Load conda
set +u
# Using miniforge3 based on user environment
source ~/miniforge3/etc/profile.d/conda.sh
export CONDA_BACKUP_CXX=${CONDA_BACKUP_CXX:-}

# Handle Git commit hash safely
if [ -d .git ]; then
    # Inside a git repo
    export GIT_COMMIT=$(git rev-parse HEAD)
else
    # Fallback if .git is missing
    export GIT_COMMIT="unknown"
fi
echo "Using commit hash: $GIT_COMMIT"

# Make temporary directories
export TMP_DIR=/data/codetr-output-dir/temp_directory_job_$SLURM_JOB_ID
mkdir -p $TMP_DIR
export MPLCONFIGDIR=$TMP_DIR/matplotlib
mkdir -p $MPLCONFIGDIR

export FONTCONFIG_PATH=/usr/share/fonts
export FONTCONFIG_CACHE=$TMP_DIR/fontconfig
export FONTCONFIG_FILE=''

export WANDB_DIR=$TMP_DIR/wandb
mkdir -p $WANDB_DIR
export WANDB_CACHE_DIR=$WANDB_DIR

# Activate environment
conda activate codetr

# Optional: print debug info
echo "Running on host: $(hostname)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi
python --version

#########################
# Run your training     #
#########################

# Ensure we are in the correct directory (shared location)
cd /data/cashel-data/Co-DETR
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_FILE="projects/configs/co_dino_vit/imagenet_vid_vit_large_co_dino.py"
WORK_DIR="/data/codetr-output-dir/work_dirs/imagenet_vid_vit_large_co_dino"

# Create work dir if it doesn't exist
mkdir -p $WORK_DIR

python -m \
  torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=29501 \
  tools/train.py \
  $CONFIG_FILE \
  --launcher pytorch \
  --work-dir $WORK_DIR
