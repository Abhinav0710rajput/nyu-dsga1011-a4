#!/bin/bash
#SBATCH --job-name=q1-train
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

# Q1 full run: fine-tunes BERT on all of IMDB train, then evaluates on IMDB test.
# Full points need more than 91% accuracy.
# Produces part1/out/ (the model checkpoint) and part1/out_original.txt (the Q1 submission file).
# Submit from the repo root:   sbatch slurm/q1_train.sbatch

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"
[ -d part1 ] || { echo "ERROR: submit this job from the repo root" >&2; exit 1; }

source slurm/env.sh

nvidia-smi
cd part1
python3 main.py --train --eval
