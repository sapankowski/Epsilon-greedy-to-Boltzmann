#!/usr/bin/env bash
#SBATCH --job-name=egtb-breakout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-cluster_manifests/breakout_1m.tsv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"
mkdir -p logs/slurm

# Activate an environment if your cluster setup provides one of these variables.
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

if [[ -n "${CONDA_SH:-}" && -n "${CONDA_ENV:-}" ]]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
TASK_LINE="$(sed -n "${LINE_NUMBER}p" "$MANIFEST")"
if [[ -z "$TASK_LINE" ]]; then
  echo "No manifest row for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 2
fi

IFS=$'\t' read -r TASK_ID CONFIG SEED STRATEGY RUN_NAME OUTPUT_DIR TOTAL_STEPS EVAL_EPISODES CHECKPOINT_INTERVAL <<< "$TASK_LINE"

echo "task_id=$TASK_ID"
echo "config=$CONFIG"
echo "seed=$SEED"
echo "strategy=$STRATEGY"
echo "run_name=$RUN_NAME"
echo "output_dir=$OUTPUT_DIR"
echo "total_steps=$TOTAL_STEPS"
echo "hostname=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

CMD=(
  "$PYTHON_BIN" -m egtb.train
  --config "$CONFIG"
  --seed "$SEED"
  --device cuda
  --total-steps "$TOTAL_STEPS"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --skip-existing
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
)

if [[ "$EVAL_EPISODES" != "default" ]]; then
  CMD+=(--eval-episodes "$EVAL_EPISODES")
fi

printf 'running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
