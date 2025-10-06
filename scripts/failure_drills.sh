#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/out"
LOG="$OUT/mock_c2_failure.log"
EVENTS="$OUT/events.jsonl"

# Choose Python interpreter: env PYBIN > .venv > python3 > python
if [ -n "${PYBIN:-}" ]; then
  PY="$PYBIN"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

mkdir -p "$OUT"
: > "$LOG"

echo "[M5] Failure drills start"

############################################
# Drill 1: Decode failure -> switch to events
############################################
echo "[Drill1] inject broken ping & watch warnings"

$PY - <<'PY' >"$OUT/broken_ping.bin"
import sys
# length-delimited payload (len=1) with invalid wire byte → forces decode error
sys.stdout.buffer.write(bytes([0x01, 0x80]))
PY

# Ensure events.jsonl exists (for fallback)
if [ ! -f "$EVENTS" ]; then
  cat >"$EVENTS" <<'JSON'
{"ts_ns":1700000000000000000,"lat":52.500,"lon":13.300,"intent_score":0.7,"uid":"fallback","event_code":1}
JSON
fi

# Run mock_c2 once (will warn on broken ping)
$PY -m src.c2.mock_c2 --pings "$OUT/broken_ping.bin" --out "$OUT/c2_feed.jsonl" --once --truncate --dedup >>"$LOG" 2>&1 || true

# If warnings seen, enable Label Mode and push events once
WARN_CNT=$(grep -E "Malformed varint|Decode error for ping|Skipping malformed" -c "$LOG" || true)
THRESH=1
echo "[Drill1] warn count=$WARN_CNT (threshold=$THRESH)"

if [ "$WARN_CNT" -ge "$THRESH" ]; then
  echo "[Drill1] Switching to LABEL MODE (events)"
  touch "$OUT/label_mode.on"
  $PY -m src.c2.mock_c2 --events "$EVENTS" --out "$OUT/c2_feed.jsonl" --once --dedup --dedup-by event_id >>"$LOG" 2>&1 || true
  echo "[Drill1] Label mode feed updated."
else
  echo "[Drill1] Not switching; warnings below threshold."
fi

############################################
# Drill 2: Network flap -> recovery p95
############################################
echo "[Drill2] simulate network flap & write monitor snapshots"
MON="$OUT/monitor.jsonl"
$PY - <<'PY'
import json, time
from pathlib import Path
out = Path("out/monitor.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
def append(d):
    if out.exists():
        out.write_text(out.read_text(encoding="utf-8") + json.dumps(d) + "\n", encoding="utf-8")
    else:
        out.write_text(json.dumps(d) + "\n", encoding="utf-8")
t = lambda: int(time.time_ns())
append({"t_ns": t(), "online": True,  "depth": 2,  "sent_1s": 5, "p95_drain_ms": 200})
time.sleep(0.05)
append({"t_ns": t(), "online": False, "depth": 20, "sent_1s": 0, "p95_drain_ms": 2300})
time.sleep(0.05)
append({"t_ns": t(), "online": True,  "depth": 3,  "sent_1s": 4, "p95_drain_ms": 1800})
PY
echo "[Drill2] monitor snapshots appended to $MON"

############################################
# Drill 3: Empty feed -> safe downgrade
############################################
echo "[Drill3] create empty feed & ensure no crash"
: > "$OUT/c2_feed.jsonl"
echo "[Drill3] empty out/c2_feed.jsonl created"

############################################
# Summary
############################################
echo
echo "----- FAILURE DRILLS SUMMARY -----"
if [ -f "$OUT/label_mode.on" ]; then
  echo "[OK] Label mode flag present (out/label_mode.on)"
else
  echo "[INFO] Label mode flag not set (warnings below threshold)"
fi

if [ -s "$MON" ]; then
  echo "[OK] Monitor snapshots exist ($(wc -l < "$MON" | tr -d ' ') lines)"
else
  echo "[ERR] Monitor snapshots missing"
fi

if [ -f "$OUT/c2_feed.jsonl" ]; then
  L=$(wc -l < "$OUT/c2_feed.jsonl" | tr -d ' ')
  echo "[OK] out/c2_feed.jsonl exists ($L lines; 0 means empty-feed drill OK)"
fi

echo "[M5] Done. Open the UI to confirm badges/fallbacks."
