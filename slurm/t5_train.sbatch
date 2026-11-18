#!/bin/bash
#SBATCH --job-name=t5-finetune
#SBATCH --account=ds_ga_1006_001-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

# Part 2: fine-tune T5-small for text-to-SQL, then write dev + test predictions.
#
# Submit from the repo root:   sbatch slurm/t5_train.sbatch
# Extra args are passed to train_t5.py, e.g. for an ablation with its own output names:
#   sbatch slurm/t5_train.sbatch --experiment_name lr1e-4 --learning_rate 1e-4
#
# With the default experiment name "experiment" it produces the Q7 submission files:
#   part2/results/t5_ft_experiment_test.sql   part2/records/t5_ft_experiment_test.pkl
# plus dev predictions (results/t5_ft_experiment_dev.sql, records/t5_ft_experiment_dev.pkl)
# and checkpoints in part2/checkpoints/ft_experiments/experiment/{best,last}/

# Use the sbatch submit dir only if it is the repo; inside an OnDemand/Jupyter
# job SLURM_SUBMIT_DIR points at the OnDemand app folder, so fall back to $PWD.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/part2" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(pwd)"
fi
cd "$REPO_DIR"
[ -d part2 ] || { echo "ERROR: submit this job from the repo root" >&2; exit 1; }

export ENV_NAME="nlp-env-part2"
source slurm/env.sh
export WANDB_MODE=disabled   # wandb is only used with --use_wandb

nvidia-smi
cd part2
python train_t5.py \
    --finetune \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --batch_size 16 \
    --test_batch_size 32 \
    "$@"
