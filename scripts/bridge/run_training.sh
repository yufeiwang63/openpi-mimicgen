#!/bin/bash

CUDA=$1
exp_name=$2
dataset_name=$3
export CUDA_VISIBLE_DEVICES=$CUDA

cd /ocean/projects/cis240052p/ywang59/openpi-mimicgen/
source .venv/bin/activate
source prepare.sh

uv run scripts/compute_norm_stats.py --config-name pi05_aloha_ours_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_aloha_ours_lora \
    --overwrite \
    --exp-name="${exp_name}" \
    --save-interval 10000

# mkdir -p /local/slurm-$SLURM_JOB_ID/local/pi_mimicgen/$dataset_name
# rsync -av /ocean/projects/cis240052p/ywang59/openpi-mimicgen/data/$dataset_name /local/slurm-$SLURM_JOB_ID/local/pi_mimicgen/$dataset_name

# uv run scripts/compute_norm_stats.py --config-name pi05_mimicgen
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_mimicgen \
#     --overwrite \
#     --exp-name="${exp_name}" \
#     --save-interval 10000

# uv run scripts/compute_norm_stats.py --config-name pi05_mimicgen_lora
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_mimicgen_lora \
#     --overwrite \
#     --exp-name="${exp_name}" \
#     --save-interval 10000
    
