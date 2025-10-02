# FieldEvidence Hackathon POC

This repository holds a proof-of-concept pipeline for detecting synthetic UAV parasite launches, emitting SALUTE micropings, and visualising them on a CoT map.

## Mission Outcome

Operator loads a synthetic UAV clip with a parasite launch. The system auto-detects the launch, emits a SALUTE Protobuf microping (≤80 B), queues if offline, and displays a CoT map pin once connectivity resumes.

## Non-Negotiable Acceptance Criteria

* ≥3 synthetic clips each yield ≥1 detected launch event.
* Microping serialized size ≤80 B (avg ≤75 B, max ≤80 B).
* Store-and-forward behaviour:

  * **Offline** → micropings saved in SQLite immediately.
  * **Online** → queue flushed ≤2 s after toggle.
* CoT pin appears in the Streamlit map ≤5 s after event detection.
* End-to-end demo runs locally on Python 3.10+ (CPU-only).

## Inputs / Outputs

### Inputs

* `.mp4` synthetic fixed-wing + parasite videos (30 fps, ≤15 s).
* Optional `.csv` labels with columns `frame,id,class,x,y,w,h`.

### Outputs

* Event log JSON Lines (`.jsonl`).
* SALUTE microping (Protobuf) using `src/wire/salute.proto` (≤80 B payload).
* CoT pin rendered in the Streamlit/Leaflet map UI.

## Quickstart (Cold-Start on Any Machine)

1. **Clone and enter project**

   ```bash
   git clone git@github.com:joonsooro/EDTH_Hamburg.git
   cd EDTH_Hamburg/field-evidence
   ```
2. **Create virtual environment + install dependencies**

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
3. **Run full guard check (proto → emit → validate)**

   ```bash
   make check
   ```

   If you see
   `✅ Pipeline check completed successfully.`
   the environment is correct and the pipeline is working.

---

## Generating Synthetic Clips

```bash
make synth1
```

or directly:

```bash
python -m src.synth.synthesize \
  --carrier data/raw/carrier.mp4 \
  --sprite  data/sprites/parasite.png \
  --out_dir data/synthetic --labels_dir data/labels \
  --clips 1 --seed 42 --sep_frame_min 110 --sep_frame_max 120 \
  --num_parasites 2 --anchor_mode wings \
  --sprite_scale_min 0.04 --sprite_scale_max 0.06 \
  --wing_dx_ratio 0.38 --wing_dy_ratio 0.06 \
  --sep_vx_range -5 5 --sep_vy_range -10 -4 --gravity 0.9 \
  --event_dist_ratio 1.1 --event_min_frames 3
```

### Notable options

* `--sprite_px_min/--sprite_px_max`: absolute pixel width range (fallback to relative scaling if 0).
* `--sprite_resize`: `nearest`, `area`, or `lanczos`.
* `--sprite_unsharp`: amount for post-resize unsharp mask (0 disables).
* `--anchor_px_offset OX OY`: fine-tune anchor position in pixels.
* `--debug 1`: writes overlay frames and diagnostics under `data/synthetic/debug/`.

## Diagnostics

* Startup prints parsed arguments and sprite sizing info.
* First three frames log tracker bbox, anchors, and sprite box `(x, y, w, h)`.
* With `--debug`, CSV diagnostics and overlay video are written to `data/synthetic/debug/`.

---

## Demo Script (≈75 s)

1. Operator plays a synthetic clip (parasite separates).
2. Detector overlays bounding boxes; divergence triggers launch event.
3. Event log and SALUTE microping preview display with byte size.
4. Network OFF → microping stored in SQLite queue.
5. Network ON → queue drains within 2 s.
6. CoT pin appears on the Streamlit map within 5 s, coloured by `event_code`.
7. Operator exports the SALUTE bundle (JSON + Proto hex preview).

---

## Make Targets

* `make setup` — create `.venv` and install dependencies.
* `make proto` — regenerate Protobuf code (`*_pb2.py`) from `src/wire/salute.proto`.
* `make synth` / `make synth1` — generate synthetic clips.
* `make detect` — run launch detection.
* `make emit` — emit SALUTE micropings (`out/pings.bin` + `ping_sizes.json`).
* `make check` — **guard check**: ensures proto generated, micropings emitted, size ≤80 B, and first messages decode correctly.
* `make replay` — placeholder for store-and-forward replay.
* `make run` — launch Streamlit UI (`src/ui/app.py`).
* `make eval` — placeholder for evaluation harness.
* `make test` — run pytest suite.

---

## Operational Guardrails

Out-of-scope: real UAV data, live ML training, crypto key management, RF integration, or production-grade UI.

---

## Requirements (Pinned)

This project relies on **pinned versions** to avoid protobuf/gRPC mismatches:

```txt
numpy==1.26.4
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
pillow==10.3.0
sqlalchemy==2.0.29
aiosqlite==0.19.0
streamlit==1.32.2
folium==0.15.1
rich==13.7.1
protobuf==5.27.2
grpcio==1.66.1
grpcio-tools==1.66.1
pytest==8.1.1
```

---

## Why the Guard Check Matters

* **Proto stubs present**: fails fast if you forgot `make proto`.
* **Microping size OK**: ensures ≤80 B contract before demo.
* **Decode OK**: guarantees varint framing + schema consistent.

Running `make check` before a demo saves you from “works on my laptop” surprises.

