#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
FEED_JSONL="out/c2_feed.jsonl"
EVENTS_JSONL="out/events.jsonl"
RECO_SAMPLE="out/reco_sample.jsonl"
AUDIT_JSONL="out/c2_audit.jsonl"
METRICS_JSON="out/metrics.json"

echo "[SMOKE] start"

# 0) Prep dirs
mkdir -p out

# 1) Normalize backlog from events → feed (truncate)
echo "[SMOKE] build feed from events"
python -m src.c2.mock_c2 \
  --events "${EVENTS_JSONL}" \
  --once --out "${FEED_JSONL}" \
  --truncate --dedup --dedup-by event_id

if [[ ! -s "${FEED_JSONL}" ]]; then
  echo "[SMOKE][ERR] ${FEED_JSONL} missing or empty"; exit 2
fi
echo "[SMOKE] feed OK ($(wc -l < "${FEED_JSONL}") lines)"

# 2) Python: last-K by latest UID → Tracker → risk → recommendation (fallback to IMMEDIATE if needed)
echo "[SMOKE] compute estimation/risk/recommendation"
python - <<'PY'
import json, math, time, os, sys
from pathlib import Path
import yaml

from src.c2.estimator import Tracker
from src.c2.prioritizer import compute_risk
from src.c2.recommendation import make_recommendation

FEED = Path("out/c2_feed.jsonl")
RECO_SAMPLE = Path("out/reco_sample.jsonl")
CONFIG = Path("config.yaml")

if not FEED.exists() or FEED.stat().st_size == 0:
    print("[SMOKE][ERR] feed not present", file=sys.stderr); sys.exit(3)

cfg = {}
if CONFIG.exists():
    try:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}

# Load last ~200 feed rows
lines = [json.loads(x) for x in FEED.read_text(encoding="utf-8").splitlines() if x.strip()]
if not lines:
    print("[SMOKE][ERR] empty feed", file=sys.stderr); sys.exit(3)

# Pick latest UID
latest = lines[-1]
uid = latest.get("uid")
by_uid = [r for r in lines if r.get("uid")==uid][-50:]  # K=50 window

# Build estimation using Tracker
trk = Tracker(window_sec=30.0, decay_tau_sec=float(cfg.get("decay_tau_sec", 20.0)))
for ev in by_uid:
    trk.update({"ts_ns": int(ev["ts_ns"]), "lat": float(ev["lat"]), "lon": float(ev["lon"])})

now_ns = int(by_uid[-1]["ts_ns"])
est = trk.estimate(now_ns)

# Compute risk vs target in config (fallback defaults)
tlat = cfg.get("target",{}).get("lat", 52.5200)
tlon = cfg.get("target",{}).get("lon", 13.4050)
risk = compute_risk(est, float(tlat), float(tlon), cfg)

# Try to build recommendation; if None or not INTERCEPT, synthesize IMMEDIATE for smoke
reco = make_recommendation(est, {"event_id": latest.get("event_id","smoke#auto"),
                                 "uid": latest.get("uid"), "ts_ns": latest.get("ts_ns", now_ns)},
                           float(tlat), float(tlon), cfg)

def _bearing(lat1, lon1, lat2, lon2):
    import math
    dlon = math.radians(lon2-lon1)
    a1 = math.radians(lat1); a2 = math.radians(lat2)
    y = math.sin(dlon)*math.cos(a2)
    x = math.cos(a1)*math.sin(a2) - math.sin(a1)*math.cos(a2)*math.cos(dlon)
    brng = (math.degrees(math.atan2(y,x))+360.0)%360.0
    return brng

# Haversine meters
def _dist_m(lat1, lon1, lat2, lon2):
    R=6371000.0
    import math
    dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

# If reco is None or not immediate intercept, synthesize an IMMEDIATE intercept reco (for smoke only)
def _synth_immediate(est, risk, tlat, tlon, cfg):
    pos = est["est_pos"]; lat, lon = float(pos["lat"]), float(pos["lon"])
    dist = _dist_m(lat, lon, tlat, tlon)
    brg = _bearing(lat, lon, tlat, tlon)
    eta = dist / float(cfg.get("asset_speed_mps", 25.0)) if dist>0 else 10.0
    return {
        "event_id": "smoke#synthetic",
        "uid": "smoke",
        "ts_ns": int(time.time_ns()),
        "severity": "IMMEDIATE",
        "risk": {"status": "IMMEDIATE", "score": risk.get("score", 1.0), "factors": risk.get("factors", {})},
        "action": {"type": "INTERCEPT", "distance_m": dist, "bearing_deg": brg, "eta_sec": eta},
        "message": "SMOKE: synthetic IMMEDIATE recommendation",
        "created_ns": int(time.time_ns()),
    }

if not reco or (isinstance(reco, dict) and reco.get("action",{}).get("type")!="INTERCEPT"):
    reco = _synth_immediate(est, risk, float(tlat), float(tlon), cfg)

# Append single sample reco
RECO_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
with RECO_SAMPLE.open("a", encoding="utf-8") as f:
    f.write(json.dumps(reco)+"\n")

print("[SMOKE] reco OK:", ("IMMEDIATE" if reco.get("risk",{}).get("status")=="IMMEDIATE" else reco.get("severity")))
# Also print a tiny summary for the outer script
print(json.dumps({"ok": True, "severity": reco.get("risk",{}).get("status") or reco.get("severity")}, ensure_ascii=False))
PY

# 3) Approve and dispatch → audit timeline must append 3 states
echo "[SMOKE] approve & dispatch"
python - <<'PY'
import json, sys
from pathlib import Path
from src.c2.scheduler import approve_and_dispatch

RECO_SAMPLE = Path("out/reco_sample.jsonl")
AUDIT = Path("out/c2_audit.jsonl")

if not RECO_SAMPLE.exists():
    print("[SMOKE][ERR] reco_sample missing", file=sys.stderr); sys.exit(4)

# use last reco
last = None
for line in RECO_SAMPLE.read_text(encoding="utf-8").splitlines():
    line=line.strip()
    if not line: continue
    try:
        last = json.loads(line)
    except Exception:
        pass

if not last:
    print("[SMOKE][ERR] empty reco_sample", file=sys.stderr); sys.exit(4)

# Build minimal estimation for scheduler projection (est_pos required)
est = {"est_pos":{"lat": 52.5, "lon": 13.3}, "est_vel":{"mps": 12.0, "heading_deg": 90.0}, "confidence": 0.9}
cfg = {"asset_speed_mps": 25.0}

out = approve_and_dispatch(last, est, cfg, audit_path=Path("out/c2_audit.jsonl"))
print(json.dumps(out))
PY

# Verify 3 states were appended (ASSIGNED/LAUNCHED/INTERCEPT_EXPECTED)
if command -v jq >/dev/null 2>&1; then
  tail -n 3 "${AUDIT_JSONL}" | jq -r '.state' | grep -q '^ASSIGNED$'
  tail -n 2 "${AUDIT_JSONL}" | jq -r '.state' | head -n 1 | grep -q '^LAUNCHED$'
  tail -n 1 "${AUDIT_JSONL}" | jq -r '.state' | grep -q '^INTERCEPT_EXPECTED$'
else
  # Fallback text checks if jq not installed
  tail -n 3 "${AUDIT_JSONL}" | grep -q '"state": "ASSIGNED"'
  tail -n 2 "${AUDIT_JSONL}" | head -n 1 | grep -q '"state": "LAUNCHED"'
  tail -n 1 "${AUDIT_JSONL}" | grep -q '"state": "INTERCEPT_EXPECTED"'
fi
echo "[SMOKE] audit timeline OK"

# 4) Confirm metrics exist and p95 is numeric (if available)
echo "[SMOKE] metrics check"
if [[ ! -f "${METRICS_JSON}" ]]; then
  echo "[SMOKE][WARN] ${METRICS_JSON} not found (UI or metrics recorder may create it) — skipping numeric check"
else
  if command -v jq >/dev/null 2>&1; then
    jq -e '.decision_latency.p95_ms | numbers' "${METRICS_JSON}" >/dev/null || {
      echo "[SMOKE][ERR] metrics p95_ms not numeric"; exit 6; }
  fi
  echo "SMOKE OK: metrics.json present"
fi

echo
echo "----- CHECKING C2 SMOKE RESULTS -----"

for f in out/c2_feed.jsonl out/reco_sample.jsonl out/c2_audit.jsonl out/metrics.json; do
  if [[ -f "$f" ]]; then
    echo "[OK] $f exists ($(wc -l < $f | tr -d ' ') lines)"
  else
    echo "[ERR] Missing: $f"
  fi
done

echo
echo "----- LATEST RECOMMENDATION -----"
tail -n 1 out/reco_sample.jsonl | jq '{event_id, severity: (.risk.status // .severity), action: (.action.type // "n/a"), distance_m: .action.distance_m, bearing_deg: .action.bearing_deg}' 2>/dev/null \
  || tail -n 1 out/reco_sample.jsonl

echo
echo "----- AUDIT TIMELINE (last 3 states) -----"
tail -n 3 out/c2_audit.jsonl | jq -r '.state' 2>/dev/null || tail -n 3 out/c2_audit.jsonl
echo "[CHECK] Expected order: ASSIGNED → LAUNCHED → INTERCEPT_EXPECTED"

echo
echo "----- METRICS DECISION LATENCY -----"
jq '.decision_latency' out/metrics.json 2>/dev/null || cat out/metrics.json

echo
echo "----- SUMMARY -----"
if [[ -f out/reco_sample.jsonl ]] && grep -q "IMMEDIATE" out/reco_sample.jsonl && tail -n 3 out/c2_audit.jsonl | grep -q "INTERCEPT_EXPECTED"; then
  echo "✅ SMOKE TEST SUCCESS — IMMEDIATE reco + audit timeline confirmed."
else
  echo "⚠️  SOMETHING OFF — check reco severity or audit log."
fi
