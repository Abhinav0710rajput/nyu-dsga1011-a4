# Shared environment setup, sourced by every SLURM script in this folder.
# Activates the `nlp-env` conda environment (or $ENV_NAME, e.g. nlp-env-part2) and points
# all caches at $SCRATCH (home directories on the cluster have a small quota).
#
# Interactive use:   source slurm/env.sh                          (Part 1 env)
#                    ENV_NAME=nlp-env-part2 source slurm/env.sh   (Part 2 env)

# Strict mode only inside scripts: in an interactive shell `set -e` would close the
# terminal on the next failing command.
case $- in
    *i*) ;;
    *) set -euo pipefail ;;
esac

# Big files go on scratch if it exists, otherwise on $HOME.
export STORAGE_ROOT="${SCRATCH:-$HOME}"

export CONDA_ROOT="${CONDA_ROOT:-$STORAGE_ROOT/miniconda3}"
export ENV_NAME="${ENV_NAME:-nlp-env}"   # Part 2 scripts set ENV_NAME=nlp-env-part2

# Caches for Hugging Face models/datasets, pip, and NLTK data.
export HF_HOME="$STORAGE_ROOT/.cache/huggingface"
export PIP_CACHE_DIR="$STORAGE_ROOT/.cache/pip"
export NLTK_DATA="$STORAGE_ROOT/nltk_data"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$NLTK_DATA"

# Load conda. Expects Miniconda installed at $CONDA_ROOT (setup_env.sbatch does that).
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found at $CONDA_ROOT. Run: sbatch slurm/setup_env.sbatch" >&2
    return 1 2>/dev/null || exit 1
fi

# Activate the env, except during setup_env.sbatch, which creates it.
if [ "${SKIP_ACTIVATE:-0}" != "1" ]; then
    _had_nounset=0; case $- in *u*) _had_nounset=1 ;; esac
    set +u   # conda activate scripts reference unset variables
    conda activate "$ENV_NAME"
    [ "$_had_nounset" = 1 ] && set -u
    echo "Activated conda env: $CONDA_DEFAULT_ENV ($(which python))"
fi
