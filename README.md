FieldEvidence 

This repository contains a complete proof-of-concept (PoC) pipeline for detecting UAV parasite launches, emitting SALUTE micropings, and performing mock C2 (command-and-control) decision-making with map and video visualization.

⸻

Mission Outcome

An operator loads a synthetic UAV clip showing a parasite launch.
The system:
	1.	Automatically detects the launch event.
	2.	Emits a serialized SALUTE Protobuf microping (≤ 80 B).
	3.	Stores it if offline and replays it when network is back online.
	4.	Generates a CoT (Cursor-on-Target) XML and displays a corresponding pin on the map.
	5.	Passes the normalized events into a mock C2 pipeline for estimation, risk scoring, and strike recommendation.

⸻

Non-Negotiable Acceptance Criteria
	•	≥ 3 synthetic clips each yield ≥ 1 detected launch event.
	•	Microping serialized size ≤ 80 B (avg ≤ 75 B, max ≤ 80 B).
	•	Store-and-forward behavior:
	•	Offline → micropings saved to SQLite immediately.
	•	Online → queue flushed ≤ 2 s after toggle.
	•	CoT pin appears in the Streamlit map ≤ 5 s after event detection.
	•	End-to-end demo runs locally on Python 3.10+ (CPU-only).

⸻

Inputs / Outputs

Inputs
	•	.mp4 synthetic fixed-wing + parasite videos (30 fps, ≤ 15 s).
	•	Optional .csv labels with columns frame,id,class,x,y,w,h.

Outputs
	•	out/events.jsonl — event log.
	•	out/pings.bin + ping_sizes.json — SALUTE micropings.
	•	out/queue.db + monitor.jsonl — store-and-forward state.
	•	out/cot/*.xml — CoT events.
	•	out/c2_feed.jsonl + out/c2_audit.jsonl — C2 pipeline outputs.
	•	Streamlit UI (video + map + C2 panel).

⸻

Quickstart (Cold-Start on Any Machine)

git clone git@github.com:joonsooro/EDTH_Hamburg.git
cd EDTH_Hamburg/field-evidence
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
make check

If you see
✅ Pipeline check completed successfully.
your environment is ready.

⸻

Generating Synthetic Clips

make synth1

or directly:

python -m src.synth.synthesize \
  --carrier data/raw/carrier.mp4 \
  --sprite  data/sprites/parasite.png \
  --out_dir data/synthetic --labels_dir data/labels \
  --clips 1 --seed 42 --sep_frame_min 110 --sep_frame_max 120 \
  --num_parasites 2 --anchor_mode wings \
  --sprite_scale_min 0.04 --sprite_scale_max 0.06

Debug options: --debug 1 saves overlays to data/synthetic/debug/.

⸻

Demo Script (≈ 90 s)
	1.	Operator plays a synthetic clip.
	2.	Detector overlays bounding boxes; parasite separation triggers an event.
	3.	Microping (≤ 80 B) preview shows hash + size.
	4.	Network OFF → microping stored in SQLite.
	5.	Network ON → queue drains ≤ 2 s.
	6.	CoT pin appears on the map within 5 s.
	7.	Mock C2 panel updates: estimation → risk → recommendation → optional “Intercept” dispatch shown on map.
	8.	Operator exports the SALUTE bundle (JSON + Proto hex).

⸻

Architecture

The system operates across four packs:

Pack	Responsibility	Representative Files
Detect + Wire (pre-C2)	Detect events from labeled frames → emit SALUTE micropings	src/detect/detect_launch.py, src/wire/emit_microping.py, src/wire/codec.py
Infra	Store-and-forward queue, connectivity simulation, CoT encoding, replay	src/infra/store_forward.py, src/infra/net_sim.py, scripts/replay_pipeline.py
C2	Receiver → estimator → risk → recommendation → scheduler (mock strike dispatch)	src/c2/mock_c2.py, src/c2/estimator.py, src/c2/prioritizer.py, src/c2/recommendation.py, src/c2/scheduler.py
UI + Ops	Streamlit UI, metrics, and Make targets	src/ui/app.py, Makefile, scripts/*.sh

End-to-End flow summary
	1.	Detect events → out/events.jsonl
	2.	Emit SALUTE pings → out/pings.bin
	3.	Replay via SQLite queue (out/queue.db) → out/sent.bin + out/monitor.jsonl
	4.	Generate CoT XML → out/cot/*.xml
	5.	C2 consumes pings/events → out/c2_feed.jsonl → estimation / risk / recommendation / dispatch → out/c2_audit.jsonl
	6.	Streamlit app visualizes map, CoT pins, metrics, and C2 panel.

Refer to ARCHITECTURE.md for deep component relationships and RUNBOOK.md for live-demo procedures.

⸻

Make Targets

Command	Purpose
make setup	Create .venv and install dependencies
make proto	Regenerate Protobuf stubs
make synth / synth1	Generate synthetic clips
make detect	Run detection
make emit	Emit SALUTE micropings
make cot	Generate CoT XMLs
make replay	Run store-and-forward replay
make run	Launch Streamlit UI
make eval	Run evaluation harness
make test	Run pytest suite
make check	Full guard check (proto → emit → validate)


⸻

Requirements (Pinned)

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


⸻

Operational Guardrails

Out-of-scope: real UAV data, live ML training, cryptographic key management, RF integration, or production-grade UI.
Use pinned versions to prevent protobuf/gRPC mismatch.

⸻

Why make check Matters
	•	Verifies Protobuf stubs exist (make proto done).
	•	Ensures microping ≤ 80 B before demo.
	•	Confirms schema integrity (decode OK).

This prevents “works on my laptop” issues before live demos.

⸻

✅ All sections now reflect the latest repository state
(inc. mock C2, config.yaml, and Streamlit C2 panel).
⸻

Credits & License

© 2025 FieldEvidence team. For hackathon demo & evaluation. Simulation-only. License TBD.

