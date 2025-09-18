#!/usr/bin/env bash
set -euo pipefail

# Resolve project root relative to this script so the script can be invoked from any directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

# Ensure a virtual environment exists for the API server runtime dependencies.
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

# Keep pip itself up to date, then install the project in editable mode so the
# `starrocks-api-server` entry point is available.
pip install --upgrade pip
pip install -e .

# Load environment variables from the user-provided .env file if it exists.
if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo "Warning: .env file not found at $ROOT_DIR/.env" >&2
fi

# Launch the API server. Using exec hands control to the uvicorn process so it
# receives signals directly (Ctrl+C, systemd stop, etc.).
exec starrocks-api-server
