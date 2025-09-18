#!/usr/bin/env bash
set -euo pipefail

# Resolve project root relative to this script so the script can be invoked from any directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

# Ensure we are operating inside the expected Conda environment. The environment name can
# be overridden by exporting STARROCKS_CONDA_ENV; otherwise we default to "mcp".
CONDA_ENV_NAME="${STARROCKS_CONDA_ENV:-mcp}"

if [ "${CONDA_DEFAULT_ENV-}" != "$CONDA_ENV_NAME" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
  else
    echo "Conda environment '$CONDA_ENV_NAME' is required but conda was not found in PATH." >&2
    exit 1
  fi
fi

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
