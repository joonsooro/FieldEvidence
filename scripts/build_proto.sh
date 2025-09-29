#!/usr/bin/env bash
set -euo pipefail

# cross-shell script path
if [ -n "${BASH_SOURCE:-}" ]; then
  _SELF="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_NAME:-}" ]; then
  _SELF="${(%):-%N}"
else
  _SELF="$0"
fi
ROOT="$(cd "$(dirname "$_SELF")/.." && pwd)"

python -m grpc_tools.protoc -I "$ROOT/src/wire" \
  --python_out="$ROOT/src/wire" \
  "$ROOT/src/wire/salute.proto"

echo "✅ Generated: src/wire/salute_pb2.py"
