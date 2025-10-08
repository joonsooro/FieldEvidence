# Field Evidence — RUNBOOK

Operational quick reference for running the demo end-to-end. Copy/paste commands are designed to work from the repo root.


## 1) Prerequisites & setup

- Python 3.11+ recommended; ffmpeg for clips; optional OpenCV for snips/viz
- Install and activate env
  - `bash scripts/activate.sh`
  - `pip install -r requirements.txt`
- (Optional) Generate protobufs (authoritative proto at `src/wire/salute.proto`)
  - `make proto`


## 2) One-command 90s demo

- Runs synth → detect → emit → CoT, then replays with 45s offline + 45s online while starting the UI
  - `bash -lc 'bash scripts/activate.sh && pip install -r requirements.txt && make synth detect emit cot && PATTERN="45:off,45:on" make replay & make run'`


## 3) Manual run sequence (event → SALUTE → offline → C2 → approval → dispatch)

- Optional: synthesize demo clips + labels
  - `make synth`
- Detect events from labels to JSONL (+ optional viz/snips)
  - `make detect`
  - Writes `out/events.jsonl`, `out/viz/*.mp4`, `out/snips/*.mp4`
- Emit SALUTE micropings (length‑delimited protobuf stream)
  - `make emit`
  - Writes `out/pings.bin`, `out/ping_sizes.json`
- Generate CoT XML from pings (for UI map/table)
  - `make cot`
  - Writes `out/cot/uid_<uid>_<tsns>.xml`
- Replay store‑and‑forward with connectivity flips (offline → online)
  - `PATTERN="30:off,30:on" make replay`
  - Updates `out/monitor.jsonl`, `out/sent.bin`, `out/queue.db`
- Start UI (Streamlit): C2 panel, map, video, metrics
  - `make run`
  - Approve recommendation in the UI → dispatch; audit goes to `out/c2_audit.jsonl`
- Headless C2 normalization (optional, no approval):
  - `python src/c2/mock_c2.py --pings out/pings.bin --out out/c2_feed.jsonl --once`


## 4) Policy tuning (config.yaml)

- File: `config.yaml` (used by UI and passed to C2 modules)
- Quick edits (requires yq) — adjust thresholds/weights and policy gate
  - `yq -Yi '.thresholds.IMMEDIATE_score=0.8 | .thresholds.SUSPECT_score=0.6' config.yaml`
  - `yq -Yi '.weights.intent=0.65 | .weights.proximity=0.35 | .weights.recent=0.25 | .weights.geo_bonus=0.2' config.yaml`
  - `yq -Yi '.policy.recommend_min_status="SUSPECT" | .policy.suspect_monitor_window_sec=30' config.yaml`
- Without yq (Python inline):
  - `python - <<'PY'
import yaml,sys
p='config.yaml'
d=yaml.safe_load(open(p))
d['thresholds'].update({'IMMEDIATE_score':0.8,'SUSPECT_score':0.6})
d['policy'].update({'recommend_min_status':'SUSPECT','suspect_monitor_window_sec':30})
open(p,'w').write(yaml.safe_dump(d,sort_keys=False))
print('updated',p)
PY`


## 5) Metrics to verify

- Payload size (p95/max) from `out/ping_sizes.json`
  - `python - <<'PY'
import json;d=json.load(open('out/ping_sizes.json'))
print('payload_p95=',d.get('p95'),'max=',d.get('max'))
PY`
- Recovery (drain) latency p95 from latest monitor snapshot (`out/monitor.jsonl`)
  - `python - <<'PY'
import json,sys
*_,last=open('out/monitor.jsonl').read().splitlines()
s=json.loads(last)
print('p95_e2e_ms=',s.get('p95_e2e_ms'),'p95_drain_ms=',s.get('p95_drain_ms'),'online=',s.get('online'))
PY`
- Decision latency (p95) from `out/metrics.json` (if UI measured decisions)
  - `python - <<'PY'
import json;d=json.load(open('out/metrics.json'))
print('decision_p95_ms=',d.get('decision_latency',{}).get('p95_ms'))
PY`


## 6) Failure drills & recovery

- Network outage → backlog then recovery
  - Outage only: `PATTERN="60:off,15:on" make replay`
  - Verify depth rises while offline then drains when online (see `out/monitor.jsonl`)
- Clear the queue safely (recover from stuck payloads)
  - `python - <<'PY'
from src.infra.store_forward import clear
clear(); print('queue cleared')
PY`
- Rebuild pings when truncated/invalid
  - `rm -f out/pings.bin && make emit`
- Microping oversize (emit exits 2): re‑emit with shorter `hash_pref` auto‑fallback already applied; if still failing, inspect top offenders
  - `python src/wire/emit_microping.py --events out/events.jsonl --out out/pings.bin --size_report out/ping_sizes.json`
- Missing ffmpeg/OpenCV (clips/snips)
  - Install ffmpeg (macOS): `brew install ffmpeg`
  - Or disable ffmpeg path: `make detect CLIP_WRITER=cv2` and install OpenCV: `pip install opencv-python`
- Path issues (run from repo root)
  - `cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"`


## 7) Adaptation kit usage

- Use your own events (JSONL → SALUTE → replay → UI)
  - `python src/wire/emit_microping.py --events <your_events.jsonl> --out out/pings.bin --size_report out/ping_sizes.json`
  - `PATTERN="10:on" make replay`  # or offline/online pattern
  - `make run`
- Use your own pings (skip detect/emit)
  - `python scripts/replay_pipeline.py --events out/pings.bin --pattern "10:on"`
  - `python src/c2/mock_c2.py --pings out/pings.bin --out out/c2_feed.jsonl --once`
- Tune detector sensitivity (labels‑only run)
  - `make detect USE_CORR=1 K_FALLBACK=2 RATIO_HI=1.0 RATIO_SLOPE_THR=0.006`
- Generate CoT only (for external tool ingestion)
  - `python scripts/emit_cot_from_pings.py --pings out/pings.bin --out_dir out/cot --max 100`


## 8) Common errors and quick fixes

- `ModuleNotFoundError: src.*`
  - Activate env or export path: `bash scripts/activate.sh` (sets `PYTHONPATH`), or `export PYTHONPATH="$(pwd)"`
- `ffmpeg not found` / clips skipped
  - Install ffmpeg or switch to `CLIP_WRITER=cv2` and `pip install opencv-python`
- `grpcio` version gate when importing `salute_pb2_grpc`
  - Not needed for this demo; if hit, install modern gRPC: `pip install grpcio>=1.66.1 grpcio-tools`
- Protobuf path mismatch
  - Use `make proto` (sources in `src/wire/salute.proto`). Avoid `scripts/build_proto.sh` unless updated to `src/wire`.
- `ValueError: Truncated length-delimited stream`
  - Recreate pings: `rm -f out/pings.bin && make emit`
- Emit exit code 2 (oversize payloads)
  - Check offenders in stderr; ensure `--size_report out/ping_sizes.json` is written; re‑emit after trimming optional fields upstream if customized.


References
- Detect: `src/detect/detect_launch.py`
- Wire: `src/wire/emit_microping.py`, `src/wire/codec.py`, `src/wire/salute.proto`
- Infra: `scripts/replay_pipeline.py`, `src/infra/store_forward.py`, `src/infra/net_sim.py`, `src/infra/cot_encode.py`
- C2: `src/c2/mock_c2.py`, `src/c2/estimator.py`, `src/c2/prioritizer.py`, `src/c2/recommendation.py`, `src/c2/scheduler.py`
- UI: `src/ui/app.py`
