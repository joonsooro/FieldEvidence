#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROTO_DIR="proto"
OUT_DIR="src"

# proto 파일 유효성 확인
test -f "$PROTO_DIR/salute.proto"

# 현재 활성화된 파이썬으로 protoc 실행
python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/salute.proto"

echo "✅ Protobuf generated into $OUT_DIR"
