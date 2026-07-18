#!/usr/bin/env bash
#
# run.sh — start the entire Terry project with one command.
#
#   ./run.sh              # set up (if needed) and start the MCP server on :9021
#   ./run.sh --port 9030  # use a different port
#   PORT=9030 ./run.sh    # same, via env var
#
# It will: create the virtualenv if missing, install dependencies, initialize the
# project (strategies/ + storage/), then launch the Terry MCP server.
#
set -euo pipefail

# --- locate the project root (this script's directory) ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PORT="${PORT:-9021}"

# --- parse --port ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "==> Terry launcher (project: $ROOT)"

# --- 1. pick a Python interpreter ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi

# --- 2. create the virtualenv if it doesn't exist ---
if [[ ! -x "$PY" ]]; then
  echo "==> Creating virtualenv at .venv"
  python3 -m venv "$VENV"
fi

# --- 3. install / update dependencies (quiet, idempotent) ---
echo "==> Installing dependencies"
"$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
"$PY" -m pip install --quiet -r "$ROOT/requirements.txt"

# --- 4. initialize project folders + sample strategy (idempotent) ---
echo "==> Initializing project"
"$PY" -m terry init --project "$ROOT" >/dev/null

# --- 5. environment check ---
"$PY" -m terry doctor --project "$ROOT" | sed 's/^/    /'

# --- 6. start the MCP server ---
echo ""
echo "==> Starting Terry MCP server on http://localhost:$PORT/mcp"
echo "    Connect an agent with:"
echo "      claude mcp add --transport http terry http://localhost:$PORT/mcp"
echo ""
exec "$PY" -m terry serve --port "$PORT" --project "$ROOT"
