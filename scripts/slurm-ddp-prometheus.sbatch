#!/bin/bash
#SBATCH --job-name=ddp-prom
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail

# Optional overrides at submit time:
#   sbatch --export=ALL,DIST_BACKEND=gloo,METRICS_PORT=9108,SCRAPE_HOLD_SECONDS=120 slurm-ddp-prometheus.sbatch
DIST_BACKEND="${DIST_BACKEND:-gloo}"
METRICS_PORT="${METRICS_PORT:-9108}"
MASTER_PORT="${MASTER_PORT:-29500}"
SCRAPE_HOLD_SECONDS="${SCRAPE_HOLD_SECONDS:-120}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
SLEEP_PER_STEP_MS="${SLEEP_PER_STEP_MS:-0}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PY_SCRIPT="${PY_SCRIPT:-${SUBMIT_DIR}/ddp_prom_metrics.py}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_ADDR MASTER_PORT DIST_BACKEND METRICS_PORT SCRAPE_HOLD_SECONDS TRAIN_STEPS SLEEP_PER_STEP_MS

echo "[$(date)] Job ${SLURM_JOB_ID}"
echo "[$(date)] Nodes: ${SLURM_JOB_NODELIST}"
echo "[$(date)] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[$(date)] Metrics on rank0: http://${MASTER_ADDR}:${METRICS_PORT}/metrics"
echo "[$(date)] Using script: ${PY_SCRIPT}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH. Load a python module first (for example: module load python3)." >&2
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Training script not found: ${PY_SCRIPT}" >&2
  echo "Submit from the directory containing ddp_prom_metrics.py, or set --export=ALL,PY_SCRIPT=/abs/path/ddp_prom_metrics.py" >&2
  exit 1
fi

srun --ntasks="${SLURM_NTASKS}" --nodes="${SLURM_JOB_NUM_NODES}" \
  python "${PY_SCRIPT}"
