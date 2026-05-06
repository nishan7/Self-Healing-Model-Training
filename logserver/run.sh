#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

module load python3

python -m pip install --user -U uv
export PATH="$HOME/.local/bin:$PATH"

pip install -r requirements.txt

ngrok http "$PORT" &

HOST=0.0.0.0 PORT="$PORT" python server.py
