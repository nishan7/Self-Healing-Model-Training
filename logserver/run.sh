#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
module load python3
PORT=8000

python3 -m pip install --user -U uv
export PATH="$HOME/.local/bin:$PATH"

python3 -m uv pip install -r requirements.txt
ngrok http "$PORT" &
HOST=0.0.0.0 PORT="$PORT" python server.py
