#!/bin/bash
#SBATCH --job-name=ddp-v5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuqs,gpuqm,gpuql
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
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

module load python3 >/dev/null 2>&1
module load ml/torch/2.6 >/dev/null 2>&1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
IFACE=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)

export MASTER_ADDR MASTER_PORT
export CKPT_PATH="$RESULTS_DIR/elastic_ckpt.pt"
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
[[ -n "$IFACE" ]] && export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"

cleanup() {
  echo "[slrum][operation] type=shuttng_down"
}
trap cleanup EXIT TERM INT

echo "[slrum][operation] type=starting"
echo "[system][startup] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR:$MASTER_PORT iface=${IFACE:-unset}"
echo "[system][paths] results_dir=$RESULTS_DIR ckpt_path=$CKPT_PATH"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash "$WORKDIR/run_node.sh" \
    "$RESULTS_DIR" \
    "$MASTER_ADDR" \
    "$MASTER_PORT" \
    "$SLURM_NNODES" \
    "$POLL_INTERVAL" \
    "$POLL_DURATION" \
    "$IB_DEVICE" \
    "$IB_PORT"
