#!/bin/bash
#SBATCH --job-name=musclemimic_training
#SBATCH --time=48:00:00
#SBATCH --account=<your_account>
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4

module load python/3.11

export PATH="$HOME/.local/bin:$PATH"

uv sync --extra smpl --extra cuda
# if using GMR
# uv sync --extra smpl --extra gmr --extra cuda

uv run fullbody/experiment.py
