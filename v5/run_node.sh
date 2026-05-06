#!/bin/bash
set -euo pipefail

RESULTS_DIR="$1"
MASTER_ADDR="$2"
MASTER_PORT="$3"
NNODES="$4"
POLL_INTERVAL="$5"
POLL_DURATION="$6"
IB_DEVICE="$7"
IB_PORT="$8"

NODE=$(hostname)

python3 telemetry_poller.py \
  --output-dir "$RESULTS_DIR" \
  --interval "$POLL_INTERVAL" \
  --duration "$POLL_DURATION" \
  --ib-device "$IB_DEVICE" \
  --ib-port "$IB_PORT" &
POLLER_PID=$!

cleanup() {
  kill "$POLLER_PID" 2>/dev/null || true
  wait "$POLLER_PID" 2>/dev/null || true
  echo "[telemetry][lifecycle] node=$NODE poller_stopped"
}
trap cleanup EXIT TERM INT

echo "[telemetry][lifecycle] node=$NODE poller_started pid=$POLLER_PID"

torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node=1 \
  --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py

echo "[system][node] node=$NODE training_finished"
