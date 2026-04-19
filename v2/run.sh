#!/bin/bash
set -e

WORKDIR="/home/018280561/Self-Healing-Model-Training/v2"
RDZV_ID="${1:-job1}"
RDZV_ENDPOINT="172.16.1.77:29500"
LOGFILE="${2:-torchrun_${RDZV_ID}.log}"

cd "$WORKDIR"

nohup bash -lc "
cd '$WORKDIR' && \
torchrun \
  --nnodes=1:2 \
  --nproc_per_node=1 \
  --max_restarts=20 \
  --rdzv_id='$RDZV_ID' \
  --rdzv_backend=c10d \
  --rdzv_endpoint='$RDZV_ENDPOINT' \
  train.py
" > "$LOGFILE" 2>&1 < /dev/null &

PID=$!
echo "Started torchrun in background"
echo "PID: $PID"
echo "Log: $WORKDIR/$LOGFILE"

disown "$PID" 2>/dev/null || true