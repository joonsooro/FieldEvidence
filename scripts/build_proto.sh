#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROTO_DIR="proto"
OUT_DIR="src"

# Validate that the proto file exists
test -f "$PROTO_DIR/salute.proto"

# Run protoc using the currently active Python
python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/salute.proto"

echo "✅ Protobuf generated into $OUT_DIR"
