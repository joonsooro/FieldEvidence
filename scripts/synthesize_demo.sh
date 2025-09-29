#!/usr/bin/env bash
set -euo pipefail
if [ -n "${BASH_SOURCE:-}" ]; then
  _SELF="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_NAME:-}" ]; then
  _SELF="${(%):-%N}"
else
  _SELF="$0"
fi
ROOT="$(cd "$(dirname "$_SELF")/.." && pwd)"
python -m src.synth.synthesize \
  --carrier "$ROOT/data/raw/carrier.mp4" \
  --sprite "$ROOT/data/sprites/parasite.png" \
  --out_dir "$ROOT/data/synthetic" \
  --labels_dir "$ROOT/data/labels" \
  --clips 5 --seed 17 --sep_frame_min 80 --sep_frame_max 200 \
  --fps 30 --max_secs 15 --blur 1 --shake 1 \
  --sprite_scale_min 0.04 --sprite_scale_max 0.08 \
  --anchor_mode wings \
  --color_match 1 --shadow_strength 0.35 \
  --sep_vx_range -3 3 --sep_vy_range -6 -2 --gravity 0.6 \
  --use_ffmpeg 0 --frames_per_clip 0
