SHELL := /bin/bash
PYTHON ?= python3
VENV := .venv
PROJECT_ROOT := $(abspath .)
ACTIVATE := source $(VENV)/bin/activate
RUN := $(ACTIVATE) && PYTHONPATH=$(PROJECT_ROOT) python
STREAMLIT := $(ACTIVATE) && PYTHONPATH=$(PROJECT_ROOT) streamlit run

# Proto paths
PROTO_SRC := src/wire/salute.proto
PROTO_DIR := $(dir $(PROTO_SRC))
PROTO_OUT := src/wire
PROTO_GEN := src/wire/salute_pb2.py src/wire/salute_pb2_grpc.py

.PHONY: setup proto synth synth1 detect emit cot replay run eval test clean check

# ---------- Environment ----------
setup: $(VENV)/bin/python
	$(ACTIVATE) && pip install --upgrade pip setuptools wheel
	$(ACTIVATE) && pip install -r requirements.txt
	# Ensure grpc tools exist (safe if already installed)
	$(ACTIVATE) && pip install -q grpcio grpcio-tools protobuf

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

# ---------- Protobuf ----------
proto: $(VENV)/bin/python $(PROTO_SRC)
	# Generate Python stubs using the venv's python
	$(RUN) -m grpc_tools.protoc \
	  -I $(PROTO_DIR) \
	  --python_out=$(PROTO_OUT) \
	  --grpc_python_out=$(PROTO_OUT) \
	  $(PROTO_SRC)
	@echo "✅ Protobuf generated into $(PROTO_OUT)"

# ---------- Data synth / detect ----------
synth: $(VENV)/bin/python
	bash scripts/synthesize_demo.sh

synth1: $(VENV)/bin/python
	$(RUN) -m src.synth.synthesize \
	  --carrier data/raw/carrier.mp4 \
	  --sprite  data/sprites/parasite.png \
	  --out_dir data/synthetic --labels_dir data/labels \
	  --clips 1 --seed 42 --sep_frame_min 110 --sep_frame_max 120 \
	  --num_parasites 2 --anchor_mode wings \
	  --sprite_scale_min 0.04 --sprite_scale_max 0.06 \
	  --wing_dx_ratio 0.38 --wing_dy_ratio 0.06 \
	  --sep_vx_range -5 5 --sep_vy_range -10 -4 --gravity 0.9 \
	  --event_dist_ratio 1.1 --event_min_frames 3

detect: $(VENV)/bin/python
	$(RUN) -m src.detect.detect_launch --mode labels \
	  --in data/synthetic --labels data/labels \
	  --out out/events.jsonl --viz out/viz \
	  --use_corr   $${USE_CORR:-0} \
	  --ratio_hi   $${RATIO_HI:-0.85} \
	  --ratio_slope_thr $${RATIO_SLOPE_THR:-0.004} \
	  --k_fallback $${K_FALLBACK:-1} \
	  --clip_out   $${CLIP_OUT:-out/snips} \
	  --clip_margin_s $${CLIP_MARGIN_S:-2.0} \
	  --clip_writer $${CLIP_WRITER:-ffmpeg} \
	  --clip_select $${CLIP_SELECT:-first}

# ---------- Wire emit ----------
emit: $(VENV)/bin/python
	$(RUN) -m src.wire.emit_microping \
	  --events out/events.jsonl \
	  --out out/pings.bin \
	  --size_report out/ping_sizes.json

# ---------- Generate CoT XML from micropings ----------
cot: $(VENV)/bin/python
	$(RUN) -m scripts.emit_cot_from_pings \
	  --pings out/pings.bin \
	  --out_dir out/cot \
	  --max $${MAX:-100}

# ---------- App / eval / test ----------
replay: $(VENV)/bin/python
	$(RUN) scripts/replay_pipeline.py --pattern "$${PATTERN:-5:off,5:on}" \
	  --limit $${LIMIT:-200} --budget_ms $${BUDGET_MS:-50}

run: $(VENV)/bin/python
	$(STREAMLIT) src/ui/app.py

eval: $(VENV)/bin/python
	@echo "Stub: implement evaluation harness in src/eval and invoke it here."

test: $(VENV)/bin/python
	$(RUN) -m pytest

clean:
	rm -f $(PROTO_OUT)/*_pb2.py $(PROTO_OUT)/*_pb2_grpc.py

check: $(VENV)/bin/python proto emit
	$(RUN) tools/check_pipeline.py

$(PROTO_GEN): $(PROTO_SRC) | $(VENV)/bin/python
	$(RUN) -m grpc_tools.protoc \
	  -I $(dir $(PROTO_SRC)) \
	  --python_out=$(dir $(PROTO_SRC)) \
	  --grpc_python_out=$(dir $(PROTO_SRC)) \
	  $(PROTO_SRC)
	@echo "✅ Protobuf generated into $(dir $(PROTO_SRC))"

proto: $(PROTO_GEN)
