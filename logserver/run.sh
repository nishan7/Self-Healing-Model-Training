#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
module load python3
PORT=8002

# Always install deps into a writable local venv (HPC nodes won't allow /usr writes)
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python3 -m pip install -U pip
python3 -m pip install -U uv
uv pip install -r requirements.txt

HOST=0.0.0.0 PORT="$PORT" python3 server.py
