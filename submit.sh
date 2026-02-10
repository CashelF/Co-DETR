#!/bin/bash
set -euo pipefail

# Sync local code to shared directory
echo "Syncing code to /data/cashel-data/Co-DETR..."
mkdir -p /data/cashel-data
rsync -av --exclude 'work_dirs' --exclude 'data' /home/cash/Co-DETR /data/cashel-data/

# Submit the job
echo "Submitting job..."
sbatch train.slurm
