#!/bin/bash
#SBATCH --job-name=setup-nlp-env-part2
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

# One-time job: creates the `nlp-env-part2` conda env for Part 2 (T5 text-to-SQL).
# Part 2 pins different package versions than Part 1, so it needs its own env.
#
# Submit from the repo root:   sbatch slurm/setup_env_part2.sbatch
# Or, on a GPU node you are already on:   bash slurm/setup_env_part2.sbatch
# Running it again is safe: steps that are already done are skipped.

# Use the sbatch submit dir only if it is the repo; inside an OnDemand/Jupyter
# job SLURM_SUBMIT_DIR points at the OnDemand app folder, so fall back to $PWD.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/part2" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(pwd)"
fi
cd "$REPO_DIR"
[ -d part2 ] || { echo "ERROR: run this from the repo root (part2/ not found in $REPO_DIR)" >&2; exit 1; }

export ENV_NAME="nlp-env-part2"
export SKIP_ACTIVATE=1
source slurm/env.sh

# 1. Create the env (conda-forge only, avoids the Anaconda ToS prompt)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Conda env '$ENV_NAME' already exists, reusing it"
else
    echo "==> Creating conda env '$ENV_NAME' (python 3.11)"
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11 pip
fi

set +u
conda activate "$ENV_NAME"
set -u
echo "==> Using python: $(which python)"

# 2. Install the Part 2 requirements (torch 2.1.2 + CUDA 12.1 wheels from PyPI)
python -m pip install --upgrade pip
python -m pip install -r part2/requirements.txt

# 3. NLTK data (load_data.py calls nltk.download('punkt') on import)
python -c "
import nltk, os
nltk.download('punkt', download_dir=os.environ['NLTK_DATA'])
"

# 4. Cache the T5-small model and tokenizer so training jobs don't download them
python -c "
from transformers import T5TokenizerFast, T5ForConditionalGeneration
T5TokenizerFast.from_pretrained('google-t5/t5-small')
T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
print('Cached google-t5/t5-small.')
"

# 5. Sanity check
nvidia-smi
python -c "
import torch, transformers
print('torch', torch.__version__, '| CUDA build', torch.version.cuda, '| transformers', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "==> Done. Environment '$ENV_NAME' is ready."
