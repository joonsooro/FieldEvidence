# README_FIX

## Setup
1. Ensure Python 3.9+ with OpenCV (`opencv-contrib-python`) and numpy available. Install from `requirements.txt` if needed: `pip install -r requirements.txt`.
2. Assets referenced by the pipeline live under `data/raw/carrier.mp4` and `data/sprites/parasite.png`.

## Generating Clips
```bash
python3 src/synth/synthesize.py \
  --carrier data/raw/carrier.mp4 \
  --sprite data/sprites/parasite.png \
  --out_dir data/synthetic \
  --labels_dir data/labels \
  --clips 1 \
  --track csrt \
  --carrier_bbox 420 310 360 140 \
  --sprite_px_min 72 --sprite_px_max 96 \
  --sprite_resize lanczos \
  --sprite_unsharp 0.6 \
  --anchor_px_offset -4 3 \
  --debug 1
```

### Notable options
- `--sprite_px_min/--sprite_px_max`: lock the sprite to an absolute width range (px). Leave at 0 to fall back to relative scaling via `--sprite_scale_min/max`.
- `--sprite_resize`: choose `nearest`, `area`, or `lanczos` resampling kernels.
- `--sprite_unsharp`: amount for post-resize unsharp mask (0 disables).
- `--sprite_premultiply`: set to 0 to disable premultiplied alpha compositing.
- `--anchor_px_offset OX OY`: fine-tune anchor positions in pixels after the wing ratio is applied.
- `--pre_sep_jitter_px`: optional wobble applied while attached (default 0 for no jitter).
- `--debug 1`: writes overlay frames to `data/synthetic/debug/` for frames {0, 30, 60, sep-2, sep, sep+2} and emits a debug video.

## Diagnostics
- Startup prints the parsed arguments, sprite source size, resize kernel, pixel sizing mode, and premultiply status.
- The first three frames log tracker bbox, computed anchors, and sprite `(x, y, w, h)`.
- When `--debug` is set, CSV diagnostics live in `data/synthetic/debug/dbg_clip_<uid>.csv` alongside the overlay video and extracted frames.
