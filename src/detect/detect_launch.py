import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-based launch detector (labels-only)")
    p.add_argument("--mode", default="labels", choices=["labels", "vision"], help="Detector mode")
    p.add_argument("--in", dest="in_dir", default="data/synthetic", help="Input synthetic clips dir")
    p.add_argument("--labels", dest="labels_dir", default="data/labels", help="Labels dir (CSV per clip)")
    p.add_argument("--out", dest="out_path", default="out/events.jsonl", help="Output JSONL path")
    p.add_argument("--viz", dest="viz_dir", default=None, help="Output viz dir (annotated videos)")

    # Legacy tuning (only meaningful when using the correlation gate)
    p.add_argument("--dist_ratio", type=float, default=1.06, help="Distance ratio to carrier width")
    p.add_argument("--corr_thr", type=float, default=0.60, help="Max abs corr threshold")
    p.add_argument("--k", type=int, default=2, help="Min consecutive frames for trigger")
    p.add_argument("--win", type=int, default=5, help="Sliding window length (frames)")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if no events found")
    p.add_argument("--corr_mode", default="vel", choices=["pos","vel"], help="correlation on position or velocity")
    p.add_argument("--corr_lag", type=int, default=0, help="positive = carrier leads parasite by this many frames")

    # New options: disable correlation gate + fallback (distance + slope)
    p.add_argument("--use_corr", type=int, default=0, help="1=use correlation gate, 0=disable (fallback only)")
    p.add_argument("--lag_sweep", type=int, default=1, help="check min corr over [-lag_sweep..+lag_sweep] when use_corr=1")
    p.add_argument("--ratio_hi", type=float, default=0.85, help="fallback: distance ratio threshold on dist/cw")
    p.add_argument("--ratio_slope_thr", type=float, default=0.004, help="fallback: min delta(dist/cw) per frame")
    p.add_argument("--k_fallback", type=int, default=1, help="fallback: consecutive frames for ratio+slope")

    # save a clip from +-2 seconds from the incident
    p.add_argument("--clip_out", default="out/snips", help="write per-uid event clip here")
    p.add_argument("--clip_margin_s", type=float, default=2.0, help="seconds before/after event")
    p.add_argument("--clip_writer", default="ffmpeg", choices=["ffmpeg","cv2"], help="clip writer backend")
    p.add_argument("--clip_select", default="best_conf", choices=["first","best_conf"],
                help="which event to export as clip (default: best_conf)")



    return p.parse_args()

def write_event_clip_ffmpeg(mp4_in: Path, t_center: float, margin: float, out_path: Path) -> None:
    """정확도를 위해 재인코딩하여 ±margin 초 클립을 저장."""
    ensure_dir(out_path.parent)
    start = max(0.0, t_center - margin)
    dur   = max(0.1, 2.0 * margin)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-i", str(mp4_in),
        "-t", f"{dur:.3f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        str(out_path),
    ]
    try:
        import subprocess
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except Exception as e:
        print(f"[warn] ffmpeg clip failed for {mp4_in} @ {t_center:.3f}s: {e}")

def write_event_clip_cv2(mp4_in: Path, t_center: float, margin: float, out_path: Path) -> None:
    """OpenCV로 ±margin 초 구간을 잘라 저장(키프레임 정밀도는 ffmpeg보다 떨어질 수 있음)."""
    ensure_dir(out_path.parent)
    try:
        import cv2
    except Exception:
        print("OpenCV not available; skipping CV2 clip:", mp4_in)
        return
    cap = cv2.VideoCapture(str(mp4_in))
    if not cap.isOpened():
        print("Unable to open video for clipping:", mp4_in); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_frame = int(max(0, round((t_center - margin) * fps)))
    end_frame   = int(min(total-1, round((t_center + margin) * fps)))
    if end_frame <= start_frame or W <= 0 or H <= 0:
        cap.release(); return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    f = start_frame
    while f <= end_frame:
        ok, frame = cap.read()
        if not ok: break
        writer.write(frame)
        f += 1
    writer.release()
    cap.release()


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return 1.0
    ma = sum(a)/n; mb = sum(b)/n
    va = sum((x-ma)*(x-ma) for x in a)
    vb = sum((y-mb)*(y-mb) for y in b)
    if va <= 1e-12 or vb <= 1e-12:
        return 1.0
    cov = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    r = cov / ((va**0.5)*(vb**0.5))
    if r != r:  # NaN guard
        return 1.0
    return r

def _sliding_corr_pos_or_vel(cx, cy, px, py, win=5, mode="vel", lag=0):
    # Apply lag (shift carrier forward/backward)
    def _lag(seq, l):
        if l == 0: return seq
        if l > 0:  return seq[l:] + [seq[-1]]*l
        l = -l;    return [seq[0]]*l + seq[:-l]

    Cx, Cy = list(cx), list(cy)
    Px, Py = list(px), list(py)
    Cx = _lag(Cx, lag); Cy = _lag(Cy, lag)

    if mode == "vel":
        Cx = [Cx[i]-Cx[i-1] for i in range(1,len(Cx))]
        Cy = [Cy[i]-Cy[i-1] for i in range(1,len(Cy))]
        Px = [Px[i]-Px[i-1] for i in range(1,len(Px))]
        Py = [Py[i]-Py[i-1] for i in range(1,len(Py))]

    n = min(len(Cx), len(Cy), len(Px), len(Py))
    Cx, Cy, Px, Py = Cx[:n], Cy[:n], Px[:n], Py[:n]
    if n == 0:
        return [1.0]

    corr = [1.0]*n
    for i in range(n):
        s = max(0, i - win + 1)
        rx = _pearson(Cx[s:i+1], Px[s:i+1])
        ry = _pearson(Cy[s:i+1], Py[s:i+1])
        # NOTE: Use the weaker correlation of the two axes
        corr[i] = min(abs(rx), abs(ry))
    # Match length (pad to original frame count)
    return [1.0]*(len(cx)-len(corr)) + corr

def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_labels_csv(path: Path) -> List[Dict[str, float]]:
    """
    Load labels CSV robustly. Support two schemas:
      - T3: frame,carrier_x,carrier_y,carrier_w,carrier_h,para_x,para_y,para_w,para_h[,event]
      - Debug: frame,carrier_cx,carrier_cy,carrier_w,carrier_h,para1_x,para1_y,para1_w,para1_h,[...]

    Returns a list of dict per frame (sorted by frame) with keys:
      frame (int), cx, cy, cw, ch, px, py
    """
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # normalize headers to lowercase for detection, but keep original keys for value access
        headers_lower = {k.lower(): k for k in reader.fieldnames or []}

        # Determine schema
        is_debug = ("carrier_cx" in headers_lower) or ("carrier_cx" in (h.lower() for h in (reader.fieldnames or [])))
        # T3 expects carrier_x and para_x
        is_t3 = ("carrier_x" in headers_lower) and ("para_x" in headers_lower)

        # Fallback: if neither detected but carrier_w present, try to infer by presence of para1_x
        if not is_debug and not is_t3:
            is_debug = ("para1_x" in headers_lower)
            if not is_debug and ("para_x" in headers_lower):
                is_t3 = True

        for r in reader:
            # frame index
            f_raw = r.get(headers_lower.get("frame", "frame"))
            f_val = _to_float(f_raw)
            if f_val is None:
                # skip malformed rows
                continue
            frame = int(round(f_val))

            # carrier box
            if is_debug:
                cx = _to_float(r.get(headers_lower.get("carrier_cx", "carrier_cx")))
                cy = _to_float(r.get(headers_lower.get("carrier_cy", "carrier_cy")))
                cw = _to_float(r.get(headers_lower.get("carrier_w", "carrier_w")))
                ch = _to_float(r.get(headers_lower.get("carrier_h", "carrier_h")))
                # parasite 1 as primary
                p1x = _to_float(r.get(headers_lower.get("para1_x", "para1_x")))
                p1y = _to_float(r.get(headers_lower.get("para1_y", "para1_y")))
                p1w = _to_float(r.get(headers_lower.get("para1_w", "para1_w")))
                p1h = _to_float(r.get(headers_lower.get("para1_h", "para1_h")))
                if None in (cx, cy, cw, ch, p1x, p1y, p1w, p1h):
                    # try T3-style parasite names if para1_* missing
                    p1x = p1x if p1x is not None else _to_float(r.get(headers_lower.get("para_x", "para_x")))
                    p1y = p1y if p1y is not None else _to_float(r.get(headers_lower.get("para_y", "para_y")))
                    p1w = p1w if p1w is not None else _to_float(r.get(headers_lower.get("para_w", "para_w")))
                    p1h = p1h if p1h is not None else _to_float(r.get(headers_lower.get("para_h", "para_h")))
                # compute parasite center from TL + w/2, h/2
                if None in (cx, cy, cw, ch, p1x, p1y, p1w, p1h):
                    # skip if insufficient
                    continue
                px = p1x + 0.5 * p1w
                py = p1y + 0.5 * p1h
            elif is_t3:
                x = _to_float(r.get(headers_lower.get("carrier_x", "carrier_x")))
                y = _to_float(r.get(headers_lower.get("carrier_y", "carrier_y")))
                cw = _to_float(r.get(headers_lower.get("carrier_w", "carrier_w")))
                ch = _to_float(r.get(headers_lower.get("carrier_h", "carrier_h")))
                px_tl = _to_float(r.get(headers_lower.get("para_x", "para_x")))
                py_tl = _to_float(r.get(headers_lower.get("para_y", "para_y")))
                pw = _to_float(r.get(headers_lower.get("para_w", "para_w")))
                ph = _to_float(r.get(headers_lower.get("para_h", "para_h")))
                if None in (x, y, cw, ch, px_tl, py_tl, pw, ph):
                    continue
                cx = x + 0.5 * cw
                cy = y + 0.5 * ch
                px = px_tl + 0.5 * pw
                py = py_tl + 0.5 * ph
            else:
                # Try best-effort mapping using present fields
                # Prefer carrier_cx/cy else compute from x/y/w/h
                cw = _to_float(r.get("carrier_w"))
                ch = _to_float(r.get("carrier_h"))
                cx = _to_float(r.get("carrier_cx"))
                cy = _to_float(r.get("carrier_cy"))
                if cx is None or cy is None:
                    x = _to_float(r.get("carrier_x"))
                    y = _to_float(r.get("carrier_y"))
                    if None in (x, y, cw, ch):
                        continue
                    cx = x + 0.5 * cw
                    cy = y + 0.5 * ch
                # parasite
                p1x = _to_float(r.get("para1_x"))
                p1y = _to_float(r.get("para1_y"))
                p1w = _to_float(r.get("para1_w"))
                p1h = _to_float(r.get("para1_h"))
                if None in (p1x, p1y, p1w, p1h):
                    p1x = _to_float(r.get("para_x"))
                    p1y = _to_float(r.get("para_y"))
                    p1w = _to_float(r.get("para_w"))
                    p1h = _to_float(r.get("para_h"))
                if None in (cx, cy, cw, ch, p1x, p1y, p1w, p1h):
                    continue
                px = p1x + 0.5 * p1w
                py = p1y + 0.5 * p1h

            rows.append({
                "frame": frame,
                "cx": float(cx),
                "cy": float(cy),
                "cw": float(cw),
                "ch": float(ch),
                "px": float(px),
                "py": float(py),
            })

    # sort by frame
    rows.sort(key=lambda d: d["frame"])
    return rows


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return 1.0  # treat as highly correlated to avoid early false triggers
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-8 or std_b < 1e-8:
        return 1.0
    r = np.corrcoef(a, b)[0, 1]
    if np.isnan(r):
        return 1.0
    return float(r)


def detect_events(
    rows: List[Dict[str, float]],
    win: int,
    corr_thr: float,
    dist_ratio: float,
    k: int,
    *,
    use_corr: int = 1,
    corr_mode: str = "vel",
    corr_lag: int = 0,
    lag_sweep: int = 1,
    ratio_hi: float = 1.08,
    ratio_slope_thr: float = 0.01,
    k_fallback: int = 2,
) -> Tuple[List[Tuple[int, float]], float, float, List[float], List[float]]:
    if not rows:
        return [], 1.0, 0.0, [], []

    cx = [r["cx"] for r in rows]
    cy = [r["cy"] for r in rows]
    px = [r["px"] for r in rows]
    py = [r["py"] for r in rows]
    cw = [r["cw"] for r in rows]
    frames = [int(r["frame"]) for r in rows]
    n = len(rows)

    # dist/cw
    ratio = []
    for i in range(n):
        dx = px[i] - cx[i]; dy = py[i] - cy[i]
        dist = (dx*dx + dy*dy) ** 0.5
        ratio.append(dist / max(cw[i], 1e-6))

    # Ratio slope (speed trend)
    dr = [0.0] + [ratio[i]-ratio[i-1] for i in range(1, n)]

    # Correlation: use the minimum over the lag sweep
    if use_corr:
        corr_all = []
        for L in range(-lag_sweep, lag_sweep+1):
            corr_L = _sliding_corr_pos_or_vel(cx, cy, px, py, win=win, mode=corr_mode, lag=corr_lag + L)
            if len(corr_L) < n:
                corr_L = [1.0]*(n - len(corr_L)) + corr_L
            corr_all.append(corr_L)
        # Per-frame minimum (absolute) correlation
        corr = [min(corr_all[j][i] for j in range(len(corr_all))) for i in range(n)]
    else:
        corr = [1.0]*n

    # Statistics
    finite_ratios = [r for r in ratio if math.isfinite(r)]
    max_ratio = max(finite_ratios) if finite_ratios else 0.0
    min_corr = min(corr) if corr else 1.0

    events: List[Tuple[int, float]] = []
    # A) Standard correlation gate
    streakA = 0; armedA = True
    # B) Fallback (distance + slope)
    streakB = 0; armedB = True

    for i in range(n):
        farA = ratio[i] > dist_ratio
        farB = ratio[i] > ratio_hi
        condA = (not use_corr) or (corr[i] < corr_thr)
        condB = dr[i] > ratio_slope_thr

        # A gate
        if farA and condA:
            streakA += 1
        else:
            streakA = 0; armedA = True
        if streakA >= k and armedA:
            conf = (1.0 - min(corr[i], 1.0)) * min(1.0, ratio[i] / max(dist_ratio, 1e-6))
            events.append((frames[i], float(conf)))
            armedA = False

        # B fallback
        if farB and condB:
            streakB += 1
        else:
            streakB = 0; armedB = True
        if streakB >= k_fallback and armedB:
            # When not using correlation, derive conf from ratio only
            conf = min(1.0, (ratio[i]-ratio_hi) / max(0.02, ratio_slope_thr*win))
            events.append((frames[i], float(conf)))
            armedB = False

    # Remove duplicate frames (prefer earlier trigger)
    events.sort(key=lambda x: x[0])
    dedup = []
    seen = set()
    for f,c in events:
        if f in seen: continue
        dedup.append((f,c)); seen.add(f)
    events = dedup

    return events, float(min_corr), float(max_ratio), corr, ratio



def load_manifest(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def annotate_video(
    mp4_path: Path,
    labels: List[Dict[str, float]],
    out_path: Path,
    trigger_frames: List[int],
    ratio_by_frame: Optional[List[float]] = None,
    corr_by_frame: Optional[List[float]] = None,
    k: int = 0,
) -> None:
    try:
        import cv2  # lazy import to avoid hard dependency if viz not requested
    except Exception:
        # If OpenCV is unavailable/broken, skip visualization gracefully
        print("OpenCV not available; skipping viz for", mp4_path)
        return
    if not mp4_path.exists():
        return
    if not labels:
        return
    ensure_dir(out_path.parent)

    # Map frame -> label row for quick access
    lab_by_frame: Dict[int, Dict[str, float]] = {int(r["frame"]): r for r in labels}
    earliest_trigger = min(trigger_frames) if trigger_frames else None

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 30.0, (width, height))

    f_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            lab = lab_by_frame.get(f_idx)
            if lab is not None:
                cx, cy = lab["cx"], lab["cy"]
                cw, ch = lab["cw"], lab["ch"]
                px, py = lab["px"], lab["py"]
                # draw carrier box (from center + size)
                x0 = int(round(cx - 0.5 * cw))
                y0 = int(round(cy - 0.5 * ch))
                x1 = int(round(cx + 0.5 * cw))
                y1 = int(round(cy + 0.5 * ch))
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # parasite center
                cv2.circle(frame, (int(round(px)), int(round(py))), 4, (0, 0, 255), -1)
            # Per-frame diagnostic overlay (top-left)
            if ratio_by_frame is not None and corr_by_frame is not None and 0 <= f_idx < len(ratio_by_frame):
                try:
                    txt = f"f={f_idx:03d}  dist/cw={ratio_by_frame[f_idx]:.3f}  corr={corr_by_frame[f_idx]:.3f}  k={k:02d}"
                except Exception:
                    txt = f"f={f_idx:03d}"
                cv2.putText(
                    frame,
                    txt,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # EVENT overlay from trigger frame
            if earliest_trigger is not None and f_idx >= earliest_trigger:
                cv2.putText(
                    frame,
                    "EVENT",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            f_idx += 1
    finally:
        writer.release()
        cap.release()


def main() -> None:
    args = parse_args()
    if args.mode == "vision":
        print("Stretch; not demo-scored")
        return

    in_dir = Path(args.in_dir)
    labels_dir = Path(args.labels_dir)
    out_path = Path(args.out_path)
    viz_dir = Path(args.viz_dir) if args.viz_dir else None

    # Prepare output
    ensure_dir(out_path.parent)
    if viz_dir is not None:
        ensure_dir(viz_dir)

    # Collect clip manifests
    manifests = sorted(in_dir.glob("clip_*.json"))
    if not manifests:
        print(f"No manifests found in {in_dir}")
    
    total_events = 0
    with out_path.open("w") as outf:
        for man_path in manifests:
            manifest = load_manifest(man_path)
            uid = manifest.get("uid") or man_path.stem
            fps = float(manifest.get("fps", 30.0))
            geo = manifest.get("geo", {})
            lat0 = float(geo.get("lat", 0.0))
            lon0 = float(geo.get("lon", 0.0))
            dpp_lat = float(geo.get("dpp_lat", 0.0))
            dpp_lon = float(geo.get("dpp_lon", 0.0))

            labels_path = labels_dir / f"{man_path.stem}.csv"
            if not labels_path.exists():
                print(f"Labels not found for {uid}: {labels_path}")
                continue
            rows = load_labels_csv(labels_path)

            events, min_corr, max_ratio, corr_by_frame, ratio_by_frame = detect_events(
            rows, win=args.win, corr_thr=args.corr_thr, dist_ratio=args.dist_ratio, k=args.k,
            use_corr=args.use_corr, corr_mode=args.corr_mode, corr_lag=args.corr_lag,
            lag_sweep=args.lag_sweep, ratio_hi=args.ratio_hi, ratio_slope_thr=args.ratio_slope_thr,
            k_fallback=args.k_fallback)


            # diagnostics per clip
            first_frame = events[0][0] if events else -1
            print(f"uid={uid} first_frame={first_frame} max_ratio={max_ratio:.3f} min_corr={min_corr:.3f}")

            # Emit JSONL per event
            for frame, conf in events:
                ev = {
                    "uid": uid,
                    "frame": int(frame),
                    "ts_ns": int((frame / fps) * 1e9),
                    "lat": lat0 + dpp_lat * frame,
                    "lon": lon0 + dpp_lon * frame,
                    "event_code": 1,
                    "conf": float(conf),
                }
                outf.write(json.dumps(ev) + "\n")
                
            total_events += len(events)

                        # Visualization (optional)
            if viz_dir is not None:
                mp4_in = in_dir / f"{man_path.stem}.mp4"
                mp4_out = viz_dir / f"{man_path.stem}.mp4"
                annotate_video(
                    mp4_in, rows, mp4_out,
                    [f for f, _ in events],
                    ratio_by_frame, corr_by_frame, k=args.k,
                )

            # === NEW: per-event subclip (only ONE per uid) ===
            # === NEW: per-event subclip (only ONE per uid) ===
            if getattr(args, "clip_out", None):
                mp4_in = in_dir / f"{man_path.stem}.mp4"
                if not mp4_in.exists():
                    print(f"[warn] source mp4 not found; skip clips: {mp4_in}")
                else:
                    chosen = None
                    if events:
                        if getattr(args, "clip_select", "best_conf") == "best_conf":
                            chosen = max(events, key=lambda e: e[1])  # choose by max conf among (frame, conf)
                        else:
                            chosen = events[0]  # earliest
                    if chosen is not None:
                        frame, conf = chosen
                        t_center = (frame / fps) if fps > 0 else 0.0
                        out_dir = Path(args.clip_out)
                        out_path = out_dir / f"{uid}_event_f{frame:04d}.mp4"
                        ensure_dir(out_dir)
                        if args.clip_writer == "ffmpeg":
                            write_event_clip_ffmpeg(mp4_in, t_center, float(args.clip_margin_s), out_path)
                        else:
                            write_event_clip_cv2(mp4_in, t_center, float(args.clip_margin_s), out_path)
                        print(f"[clip] wrote {out_path} (frame={frame}, conf={conf:.3f}, t≈{t_center:.2f}s)")




    # strict mode exit
    if args.strict and total_events == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
