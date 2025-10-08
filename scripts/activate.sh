# scripts/activate.sh
# zsh / bash compatible; use minimal options to avoid exiting the shell

# Resolve script file path
if [ -n "${BASH_SOURCE:-}" ]; then
  _SELF="${BASH_SOURCE[0]}"       # bash
elif [ -n "${ZSH_NAME:-}" ]; then
  _SELF="${(%):-%N}"              # zsh
else
  _SELF="$0"                      # fallback
fi

# Repo root
ROOT="$(cd "$(dirname "$_SELF")/.." && pwd)"

# Choose Python if available: prefer 3.11
if command -v python3.11 >/dev/null 2>&1; then
  PYBIN="python3.11"
else
  PYBIN="python3"
fi

# Create venv if missing
if [ ! -d "$ROOT/.venv" ]; then
  "$PYBIN" -m venv "$ROOT/.venv"
fi

# Activate
# shellcheck disable=SC1090
. "$ROOT/.venv/bin/activate"

# Paths
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

echo "✅ venv on: $(python -V) | PYTHONPATH=$PYTHONPATH"
