#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

module load python

python -m pip install --user -U uv
export PATH="$HOME/.local/bin:$PATH"

uv pip install -r requirements.txt

PORT=8000
if [[ "${NGROK:-0}" == "1" ]]; then
  ngrok http "$PORT" &
fi

HOST=0.0.0.0 PORT="$PORT" python server.py
