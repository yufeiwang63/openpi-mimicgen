#!/bin/bash

CUDA=$1
exp_name=$2
dataset_name=$3
export CUDA_VISIBLE_DEVICES=$CUDA


cd /project/flame/yufeiw2/openpi-mimicgen
source .venv/bin/activate
source prepare.sh

mkdir -p /tmp/data/$dataset_name
gcloud storage cp -r gs://cmu-gpucloud-yufeiw2/mimicgen_pi/$dataset_name/ /tmp/data/

# uv run scripts/compute_norm_stats.py --config-name pi05_mimicgen
# export CUDA_VISIBLE_DEVICES=$CUDA
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_mimicgen \
#     --checkpoint_base_dir /tmp \
#     --overwrite \
#     --exp-name="${exp_name}" \
#     --save-interval 10000

uv run scripts/compute_norm_stats.py --config-name pi05_mimicgen_lora
export CUDA_VISIBLE_DEVICES=$CUDA
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_mimicgen_lora \
    --checkpoint_base_dir /tmp \
    --overwrite \
    --exp-name="${exp_name}" \
    --save-interval 10000
    
