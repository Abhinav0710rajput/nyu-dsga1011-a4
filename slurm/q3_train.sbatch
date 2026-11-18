#!/bin/bash
#SBATCH --job-name=q3-augmented
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

# Q3: trains BERT on the augmented data (25k original + 5k transformed), then evaluates on both test sets.
# Produces part1/out_augmented/ (the model),
#          part1/out_augmented_original.txt and part1/out_augmented_transformed.txt (the Q3 submission files).
# Submit from the repo root:   sbatch slurm/q3_train.sbatch

# Use the sbatch submit dir only if it is the repo; inside an OnDemand/Jupyter
# job SLURM_SUBMIT_DIR points at the OnDemand app folder, so fall back to $PWD.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/part1" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(pwd)"
fi
cd "$REPO_DIR"
[ -d part1 ] || { echo "ERROR: submit this job from the repo root" >&2; exit 1; }

source slurm/env.sh

nvidia-smi
cd part1

# Train on the augmented data, then evaluate on the transformed test set (handout command)
python3 main.py --train_augmented --eval_transformed

# Evaluate the augmented model on the original test set
python3 main.py --eval --model_dir out_augmented
