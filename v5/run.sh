#!/bin/bash
#SBATCH --job-name=ddp-v5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuqs,gpuqm,gpuql
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=TERM@60

set -euo pipefail

# ---- Prototype knobs (edit directly) ----
WORKDIR="/home/018280561/Self-Healing-Model-Training/v5"
MASTER_PORT=29500
POLL_INTERVAL=0.5
POLL_DURATION=86400
IB_DEVICE="mlx4_0"
IB_PORT=1
# -----------------------------------------

cd "$WORKDIR"
mkdir -p logs

RESULTS_DIR="$WORKDIR/results/$SLURM_JOB_ID"
mkdir -p "$RESULTS_DIR"

module load python3
module load ml/torch/2.6

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
IFACE=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)

export MASTER_ADDR MASTER_PORT
export CKPT_PATH="$RESULTS_DIR/elastic_ckpt.pt"
export NCCL_IB_DISABLE=1
[[ -n "$IFACE" ]] && export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"

cleanup() {
  [[ -n "${POLLER_PID:-}" ]] && kill "$POLLER_PID" 2>/dev/null || true
  wait "${POLLER_PID:-}" 2>/dev/null || true
  echo "[slrum][operation] type=shuttng_down"
}
trap cleanup EXIT TERM INT

echo "[slrum][operation] type=starting"
echo "[system][startup] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR:$MASTER_PORT iface=${IFACE:-unset}"
echo "[system][paths] results_dir=$RESULTS_DIR ckpt_path=$CKPT_PATH"

srun --label --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  python3 telemetry_poller.py \
    --output-dir "$RESULTS_DIR" \
    --interval "$POLL_INTERVAL" \
    --duration "$POLL_DURATION" \
    --ib-device "$IB_DEVICE" \
    --ib-port "$IB_PORT" &
POLLER_PID=$!

echo "[system][telemetry] poller_step_pid=$POLLER_PID"

srun --label --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=1 \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train.py
