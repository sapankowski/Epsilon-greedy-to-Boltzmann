#!/usr/bin/env bash
#SBATCH --job-name=egtb-atari-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"
mkdir -p logs/slurm

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

if [[ -n "${CONDA_SH:-}" && -n "${CONDA_ENV:-}" ]]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

"$PYTHON_BIN" -m egtb.train \
  --config configs/atari/breakout_annealed.yaml \
  --seed 0 \
  --device cuda \
  --total-steps 100000 \
  --eval-episodes 2 \
  --output-dir runs/atari_smoke \
  --run-name breakout_annealed_smoke_seed0 \
  --skip-existing \
  --checkpoint-interval 50000
