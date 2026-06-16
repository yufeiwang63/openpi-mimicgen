#!/bin/bash


#SBATCH -N 1 # Number of nodes
#SBATCH -n 6
#SBATCH -p ROBO
#SBATCH --gpus=h100:1 #GPU specification. H100
#SBATCH -t 100:00:00 # Estimated time, 48hour max. DD-HH:MM.
#SBATCH --job-name pi-fold-onesie
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeiw2@andrew.cmu.edu
#SBATCH --exclude=robo-gh001


# use the bash shell
set -x 
# echo each command to standard out before running it

#### start training
sleep 1000000000

# cd /ocean/projects/cis240052p/ywang59/openpi-mimicgen
# source .venv/bin/activate
# source prepare.sh
# uv run scripts/compute_norm_stats.py --config-name pi05_aloha_ours_lora
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_aloha_ours_lora --exp-name=pi05_aloha_ours_lora_joint --overwrite
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_aloha_ours_lora --exp-name=pi05_aloha_ours_lora_6d_train_longer --overwrite