#!/bin/bash
#SBATCH --job-name=setup-nlp-env
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out

# One-time job: installs Miniconda (if needed), creates the `nlp-env` conda env,
# installs the Part 1 requirements, downloads the NLTK data, and caches
# bert-base-cased plus the IMDB dataset.
#
# Submit from the repo root:   sbatch slurm/setup_env.sbatch
# Running it again is safe: steps that are already done are skipped.

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"
[ -d part1 ] || { echo "ERROR: submit this job from the repo root (part1/ not found in $REPO_DIR)" >&2; exit 1; }

export SKIP_ACTIVATE=1
CONDA_ROOT="${CONDA_ROOT:-${SCRATCH:-$HOME}/miniconda3}"

# 1. Install Miniconda if missing
if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] && ! command -v conda >/dev/null 2>&1; then
    echo "==> Installing Miniconda to $CONDA_ROOT"
    INSTALLER="$(mktemp -d)/miniconda.sh"
    URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    if command -v wget >/dev/null 2>&1; then wget -q "$URL" -O "$INSTALLER"; else curl -fsSL "$URL" -o "$INSTALLER"; fi
    bash "$INSTALLER" -b -p "$CONDA_ROOT"
    rm -f "$INSTALLER"
fi

source slurm/env.sh

# 2. Create the env from conda-forge only, so Anaconda's default-channel terms-of-service prompt can't block it
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

# 3. Install the Part 1 requirements. On Linux, torch 2.1.2 from PyPI ships with CUDA 12.1, which supports the A100.
python -m pip install --upgrade pip
python -m pip install -r part1/requirements.txt

# 4. Download the NLTK data used by part1/utils.py
python -c "
import nltk, os
d = os.environ['NLTK_DATA']
for pkg in ['wordnet', 'omw-1.4', 'punkt']:
    nltk.download(pkg, download_dir=d)
"

# 5. Cache the model, tokenizer, IMDB dataset, and accuracy metric, so training jobs don't need to download them
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
AutoTokenizer.from_pretrained('bert-base-cased')
AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
load_dataset('imdb')
evaluate.load('accuracy')
print('Cached bert-base-cased, imdb, accuracy metric.')
"

# 6. Check that torch can see the GPU
nvidia-smi
python -c "
import torch, transformers, datasets
print('torch', torch.__version__, '| CUDA build', torch.version.cuda)
print('transformers', transformers.__version__, '| datasets', datasets.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "==> Done. Environment '$ENV_NAME' is ready."
