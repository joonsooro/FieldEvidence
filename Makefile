SHELL := /bin/bash
PYTHON ?= python3
VENV := .venv
PROJECT_ROOT := $(abspath .)
ACTIVATE := source $(VENV)/bin/activate
RUN := $(ACTIVATE) && PYTHONPATH=$(PROJECT_ROOT) python
STREAMLIT := $(ACTIVATE) && PYTHONPATH=$(PROJECT_ROOT) streamlit run
PROTO_SRC := src/wire/salute.proto
PROTO_OUT := src/wire

.PHONY: setup synth detect proto emit replay run eval test

setup: $(VENV)/bin/python
	$(ACTIVATE) && pip install --upgrade pip setuptools wheel
	$(ACTIVATE) && pip install -r requirements.txt

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

proto: $(VENV)/bin/python $(PROTO_SRC)
	bash scripts/build_proto.sh

synth: $(VENV)/bin/python
	bash scripts/synthesize_demo.sh

synth1: $(VENV)/bin/python
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

detect:
	python -m src.detect.detect_launch --mode labels \
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


emit: $(VENV)/bin/python
	@echo "Stub: implement SALUTE microping emitter in src/infra and invoke it here."

replay: $(VENV)/bin/python
	@echo "Stub: implement queued microping replay pipeline and invoke it here."

run: $(VENV)/bin/python
	$(STREAMLIT) src/ui/app.py

eval: $(VENV)/bin/python
	@echo "Stub: implement evaluation harness in src/eval and invoke it here."

test: $(VENV)/bin/python
	$(RUN) -m pytest
