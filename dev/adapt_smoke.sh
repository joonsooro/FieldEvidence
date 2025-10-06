#!/usr/bin/env bash
set -euo pipefail

mkdir -p out

echo "[ADAPT] simple_json → out/events.jsonl"
python tools/adapter.py --profile simple_json --in dev/samples/vendor_simple.jsonl --out out/events.jsonl

echo "[ADAPT] append nested_json"
python tools/adapter.py --profile nested_json --in dev/samples/vendor_nested.jsonl --out out/events.jsonl

echo "[ADAPT] append csv_v1"
python tools/adapter.py --profile csv_v1 --in dev/samples/vendor_csv.csv --out out/events.jsonl

wc -l out/events.jsonl

echo "[ADAPT] run through mock_c2 once"
python -m src.c2.mock_c2 --events out/events.jsonl --once --out out/c2_feed.jsonl --truncate --dedup --dedup-by event_id
wc -l out/c2_feed.jsonl

echo "[ADAPT] OK"

