# FieldEvidence Hackathon POC

This repository holds a proof-of-concept pipeline for detecting synthetic UAV parasite launches, emitting SALUTE micropings, and visualising them on a CoT map.

## Mission Outcome
Operator loads a synthetic UAV clip with a parasite launch. The system auto-detects the launch, emits a SALUTE Protobuf microping (≤80 B), queues if offline, and displays a CoT map pin once connectivity resumes.

## Non-Negotiable Acceptance Criteria
- ≥3 synthetic clips each yield ≥1 detected launch event.
- Microping serialized size ≤80 B (avg ≤75 B, max ≤80 B).
- Store-and-forward behaviour:
  - **Offline** → micropings saved in SQLite immediately.
  - **Online** → queue flushed ≤2 s after toggle.
- CoT pin appears in the Streamlit map ≤5 s after event detection.
- End-to-end demo runs locally on Python 3.10+ (CPU-only).

## Inputs / Outputs
### Inputs
- `.mp4` synthetic fixed-wing + parasite videos (30 fps, ≤15 s).
- Optional `.csv` labels with columns `frame,id,class,x,y,w,h`.

### Outputs
- Event log JSON Lines (`.jsonl`).
- SALUTE microping (Protobuf) using `src/wire/salute.proto` (≤80 B payload).
- CoT pin rendered in the Streamlit/Leaflet map UI.

## Quickstart
1. **Activate tooling**
   ```bash
   ./scripts/activate.sh
   pip install -r requirements.txt   # if first run
   ```
   Alternatively, run `make setup` to provision the virtual environment and install dependencies.
2. **Generate synthetic clips** — `make synth` *(placeholder: implement generator in `src/synth/`).*
3. **Compile protobufs** — `make proto` to refresh Python stubs in `src/wire/`.
4. **Run detection** — `make detect` *(placeholder: implement detector entry point in `src/detect/`).*
5. **Emit micropings** — `make emit` *(placeholder: implement queue/transport in `src/infra/`).*
6. **Replay queued events** — `make replay` *(placeholder: implement store-and-forward behaviour).* 
7. **Launch UI** — `make run` to start the Streamlit dashboard (expected app at `src/ui/app.py`).
8. **Evaluate & test** — `make eval` *(placeholder harness in `eval/`)* and `make test` for the pytest suite.

## Demo Script (≈75 s)
1. Operator plays a synthetic clip (parasite separates).
2. Detector overlays bounding boxes; divergence triggers launch event.
3. Event log and SALUTE microping preview display with byte size.
4. Network toggled OFF → microping stored in SQLite queue.
5. Network toggled ON → queue drains within 2 s.
6. CoT pin appears on the Streamlit map within 5 s, coloured by `event_code`.
7. Operator exports the SALUTE bundle (JSON + Proto hex preview).

## Make Targets
- `make setup` — create/refresh `.venv` and install dependencies.
- `make proto` — regenerate Protobuf code from `src/wire/salute.proto`.
- `make synth` — placeholder for synthetic clip creation workflow.
- `make detect` — placeholder for launch detection pipeline.
- `make emit` — placeholder for SALUTE microping emitter.
- `make replay` — placeholder for store-and-forward replay.
- `make run` — launch Streamlit UI (`src/ui/app.py`).
- `make eval` — placeholder for evaluation harness.
- `make test` — run pytest suite.

## Operational Guardrails
Out-of-scope activities: real UAV data collection, live ML training, crypto key management, RF/radio integration, or advanced production UI work.
