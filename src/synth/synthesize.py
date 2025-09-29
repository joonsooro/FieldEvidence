"""Synthetic clip generator for parasite launch scenarios."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Set

import numpy as np

import cv2

try: 
    legacy = cv2.legacy if hasattr(cv2, "legacy") else None 
except Exception: 
    legacy = None

ANCHOR_MODES = {"wings", "top_center", "center"}


@dataclass
class ClipSummary:
    uid: str
    sep_frame: int
    event_frame: int
    fps: int
    width: int
    height: int
    sprite_px: Tuple[int, int]
    parasites: int
    anchor_mode: str
    writer_kind: str
    out_path: Path

def safe_composite_rgba_bgr(frame_bgr: np.ndarray,
                            sprite_rgba: np.ndarray,
                            x: int, y: int) -> None:
    """
    (리버트) 단순 알파 블렌딩. 프리멀티플라이/샤픈 없음.
    frame_bgr: BGR uint8, sprite_rgba: RGBA 또는 BGRA.
    """
    H, W = frame_bgr.shape[:2]
    h, w = sprite_rgba.shape[:2]
    if h <= 0 or w <= 0:
        return

    # 프레임 클리핑
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return

    sx0 = x0 - x; sy0 = y0 - y
    sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)

    roi = frame_bgr[y0:y1, x0:x1]
    spr = sprite_rgba[sy0:sy1, sx0:sx1, ...]

    # 3채널이면 임시 알파 생성
    if spr.shape[2] == 3:
        gray = cv2.cvtColor(spr, cv2.COLOR_BGR2GRAY)
        _, a = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        spr = np.dstack([spr, a.astype(np.uint8)])
    elif spr.shape[2] != 4:
        raise ValueError("sprite must have 3 or 4 channels")

    # RGBA → BGR + A
    rgb = spr[:, :, :3]
    alpha = spr[:, :, 3].astype(np.float32) / 255.0
    bgr = rgb[..., ::-1].astype(np.float32)

    # 표준 알파 블렌딩
    roi_f = roi.astype(np.float32)
    out = bgr * alpha[..., None] + roi_f * (1.0 - alpha[..., None])
    roi[:] = np.clip(out, 0, 255).astype(np.uint8)


def wing_anchors(cx: float, cy: float, w: float, h: float,
                 dxr: float = 0.35, dyr: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute left and right wing anchor centers (float precision)."""
    left = (cx - dxr * w, cy + dyr * h)
    right = (cx + dxr * w, cy + dyr * h)
    return left, right


class Parasite:
    """Stateful parasite payload with simple two-state motion."""

    def __init__(self, x: float, y: float, w: int, h: int) -> None:
        self.x = float(x)
        self.y = float(y)
        self.w = int(w)
        self.h = int(h)
        self.vx = 0.0
        self.vy = 0.0
        self.detached = False
        self.detach_frame: Optional[int] = None
        self.k = 0
        self.first_event: Optional[int] = None
        self.sprite: Optional[np.ndarray] = None
        self.first_size: Optional[Tuple[int, int]] = None
        self.active = True
        self.scale: Optional[float] = None
        self.abs_width: Optional[int] = None
        self.last_width_px: Optional[int] = None
        self.anchor_center: Tuple[float, float] = (float(x), float(y))


class Synthesizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        np.random.seed(args.seed)

        vx_min, vx_max = args.sep_vx_range
        if vx_min > vx_max:
            vx_min, vx_max = vx_max, vx_min
        vy_min, vy_max = args.sep_vy_range
        if vy_min > vy_max:
            vy_min, vy_max = vy_max, vy_min
        self.sep_vx_range = (float(vx_min), float(vx_max))
        self.sep_vy_range = (float(vy_min), float(vy_max))

        self.num_parasites = max(1, min(int(args.num_parasites), 2))
        self.anchor_mode = args.anchor_mode if args.anchor_mode in ANCHOR_MODES else "wings"
        self.wing_dx_ratio = args.wing_dx_ratio
        self.wing_dy_ratio = args.wing_dy_ratio

        # 🔙 리버트: 새 파라미터 제거, 단순 기본값만 유지
        self.pre_sep_jitter_px = 0
        self.anchor_px_offset = (0.0, 0.0)
        self.sprite_px_min = 0
        self.sprite_px_max = 0
        self.sprite_resize_kind = "area"
        self.sprite_unsharp = 0.0
        self.sprite_premultiply = 0

        self.event_dist_ratio = float(args.event_dist_ratio)
        self.event_min_frames = max(1, int(args.event_min_frames))
        self.gravity = float(args.gravity)
        self.debug = bool(getattr(args, "debug", 0))
        self.color_match = bool(getattr(args, "color_match", 1))
        self.shadow_strength = float(getattr(args, "shadow_strength", 0.35))

        self.out_dir = Path(args.out_dir)
        self.labels_dir = Path(args.labels_dir)
        self.debug_dir = self.out_dir / "debug"
        self.frames_per_clip = args.frames_per_clip
        self.keep_size = bool(getattr(args, "keep_size", 0))

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        if self.frames_per_clip > 0:
            (self.out_dir / "frames").mkdir(parents=True, exist_ok=True)

        self.sprite_rgba = self._load_sprite(Path(args.sprite))
        self.carrier_frames, self.carrier_fps = self._load_carrier(Path(args.carrier))
        self.track_mode = getattr(args, "track", "none")
        self.init_bbox = tuple(args.carrier_bbox) if getattr(args, "carrier_bbox", None) else None
        self.use_roi = bool(getattr(args, "roi", False))
        self._summary_printed = False

    def _resolve_resize_interp(self, kind: str) -> int:
        kind_l = (kind or "lanczos").lower()
        mapping = {
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        if kind_l not in mapping:
            print(f"[warn] unknown resize kernel '{kind}', defaulting to lanczos", file=sys.stderr)
        return mapping.get(kind_l, cv2.INTER_LANCZOS4)

    def _load_sprite(self, sprite_path: Path) -> np.ndarray:
        if not sprite_path.exists():
            raise FileNotFoundError(f"Missing sprite: {sprite_path}")
        spr = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
        if spr is None:
            raise RuntimeError(f"Failed to load sprite: {sprite_path}")
        if spr.ndim != 3:
            raise ValueError("Sprite must be HxWxC array")
        if spr.shape[2] == 3:
            gray = cv2.cvtColor(spr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            spr = np.dstack([spr, mask])
        elif spr.shape[2] != 4:
            raise ValueError("Sprite must have 3 or 4 channels")
        # (리버트) BGRA 그대로 사용
        self.sprite_source_size = (spr.shape[1], spr.shape[0])
        return spr.astype(np.uint8)

    def _load_carrier(self, carrier_path: Path) -> Tuple[List[np.ndarray], float]:
        if not carrier_path.exists():
            raise FileNotFoundError(f"Missing carrier clip: {carrier_path}")
        cap = cv2.VideoCapture(str(carrier_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open carrier video: {carrier_path}")
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or self.args.fps
        raw_frames: List[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                raw_frames.append(frame)
        finally:
            cap.release()
        if not raw_frames:
            raise RuntimeError("Carrier clip has no frames")
        target_fps = self.args.fps
        step = max(int(round(orig_fps / target_fps)) if orig_fps else 1, 1)
        frames = raw_frames[::step]
        max_frames = int(target_fps * self.args.max_secs)
        frames = frames[:max_frames]
        if not frames:
            frames = raw_frames[:max_frames or len(raw_frames)]
        height, width = frames[0].shape[:2]
        if not self.keep_size and height > 720:
            scale = 720 / height
            new_size = (int(round(width * scale)), 720)
            frames = [cv2.resize(f, new_size, interpolation=cv2.INTER_AREA) for f in frames]
            width, height = new_size
        return frames, target_fps

    def _print_pipeline_summary(self, ref_bbox: Tuple[float, float, float, float]) -> None:
        if self._summary_printed:
            return
        args_payload = {k: getattr(self.args, k) for k in vars(self.args)}
        print(f"[cfg] args={args_payload}")
        cw = ref_bbox[2]
        frame_w = self.carrier_frames[0].shape[1]
        max_frame = max(8, frame_w - 4)
        if self.sprite_px_min > 0 or self.sprite_px_max > 0:
            low = self.sprite_px_min if self.sprite_px_min > 0 else self.sprite_px_max
            high = self.sprite_px_max if self.sprite_px_max > 0 else self.sprite_px_min
            low = int(np.clip(low, 8, max_frame))
            high = int(np.clip(high, 8, max_frame))
            if high < low:
                low, high = high, low
            target_descr = f"[{low}, {high}]px (absolute)"
        else:
            min_px = int(round(cw * self.args.sprite_scale_min))
            max_px = int(round(cw * self.args.sprite_scale_max))
            target_descr = f"[{min_px}, {max_px}]px (relative on cw={cw:.1f})"
        print(
            f"[cfg] resize_kernel={self.sprite_resize_kind} sprite_src={self.sprite_source_size} "
            f"target_px={target_descr} premult={self.sprite_premultiply}"
        )
        self._summary_printed = True

    def generate(self) -> None:
        summaries: List[ClipSummary] = []
        carrier_bboxes = self._estimate_carrier_track(self.carrier_frames)

        for clip_idx in range(self.args.clips):
            clip_uid = f"clip_{clip_idx:02d}"
            sep_frame = self.rng.randint(self.args.sep_frame_min, self.args.sep_frame_max)
            sep_frame = min(sep_frame, len(self.carrier_frames) - 2)
            summary = self._render_clip(
                clip_uid=clip_uid,
                sep_frame=sep_frame,
                carrier_bboxes=carrier_bboxes,
            )
            summaries.append(summary)

        vx_min, vx_max = self.sep_vx_range
        vy_min, vy_max = self.sep_vy_range
        for summary in summaries:
            print(
                f"{summary.uid} sep={summary.sep_frame} event={summary.event_frame} "
                f"fps={summary.fps} size={summary.width}x{summary.height} "
                f"parasites={summary.parasites} anchors={summary.anchor_mode} "
                f"sprite={summary.sprite_px[0]}x{summary.sprite_px[1]} writer={summary.writer_kind} "
                f"ratio={self.event_dist_ratio:.2f} K={self.event_min_frames} "
                f"vx={{{vx_min:.2f},{vx_max:.2f}}} vy={{{vy_min:.2f},{vy_max:.2f}}} "
                f"g={self.gravity:.2f} path={summary.out_path}"
            )


    def _estimate_carrier_track(self, frames: Sequence[np.ndarray]) -> List[Tuple[float,float,float,float]]:
        """Return (cx,cy,w,h) per frame. If CSRT enabled, run pixel tracker; else fallback heuristic."""
        if self.track_mode != "csrt":
            # 기존 휴리스틱(스무딩 포함)
            smoothed = []
            prev = None
            for frame in frames:
                bbox = self._estimate_carrier_bbox(frame)
                if prev is None:
                    prev = bbox
                else:
                    prev = tuple(prev[i]*0.8 + bbox[i]*0.2 for i in range(4))  # type: ignore
                smoothed.append(prev)  # type: ignore[arg-type]
            return smoothed

        # --- CSRT 경로 ---
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            tracker_ctor = legacy.TrackerCSRT_create
        elif hasattr(cv2, "TrackerCSRT_create"):
            tracker_ctor = cv2.TrackerCSRT_create
        else:
            raise RuntimeError("OpenCV CSRT tracker not available. Install opencv-contrib-python.")

        # 1) 첫 프레임 준비
        first = frames[0].copy()
        H, W = first.shape[:2]

        # 2) 초기 bbox 결정
        if self.use_roi:
            # GUI 선택(헤드리스면 사용 금지)
            sel = cv2.selectROI("select carrier", first, False, False)  # returns (x,y,w,h)
            cv2.destroyWindow("select carrier")
            if sel is None or sel[2] <= 0 or sel[3] <= 0:
                raise ValueError("ROI selection cancelled/invalid")
            init_xywh = sel
        elif self.init_bbox is not None:
            x, y, w, h = self.init_bbox
            # 화면 안으로 클램프
            x = max(0, min(x, W-1))
            y = max(0, min(y, H-1))
            w = max(8, min(w, W - x))
            h = max(8, min(h, H - y))
            init_xywh = (int(x), int(y), int(w), int(h))
        else:
            # ROI 미지정 → 휴리스틱으로 첫 bbox 잡고 사용
            cx, cy, bw, bh = self._estimate_carrier_bbox(first)
            init_xywh = (int(cx - bw/2), int(cy - bh/2), int(bw), int(bh))

        # 3) 트래커 생성/초기화
        tracker = tracker_ctor()
        ok = tracker.init(first, init_xywh)
        if not ok:
            raise RuntimeError("CSRT init failed")

        # 4) 프레임 순회하며 xywh → cx,cy,w,h로 변환
        out: List[Tuple[float,float,float,float]] = []
        for i, f in enumerate(frames):
            if i == 0:
                x, y, w, h = init_xywh
            else:
                ok, box = tracker.update(f)
                if not ok:
                    # 실패 시 직전 박스 유지 + 약한 확장(드리프트 방어)
                    x, y, w, h = x, y, w, h
                else:
                    x, y, w, h = box
            cx = x + w/2.0
            cy = y + h/2.0
            out.append((cx, cy, float(w), float(h)))
        return out


    def _estimate_carrier_bbox(self, frame: np.ndarray) -> Tuple[float, float, float, float]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_not(thresh)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        cx, cy = w / 2, h / 2
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < (w * h) * 0.01:
                continue
            cxx = x + cw / 2
            cyy = y + ch / 2
            if abs(cxx - cx) > w * 0.25 or abs(cyy - cy) > h * 0.25:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, cw, ch)
        if best is None:
            default_w = w * 0.4
            default_h = h * 0.2
            best = (
                cx - default_w / 2,
                cy - default_h / 2,
                default_w,
                default_h,
            )
        x0, y0, bw, bh = best
        center_x = x0 + bw / 2
        center_y = y0 + bh / 2
        return (center_x, center_y, bw, bh)

    def _render_clip(
        self,
        clip_uid: str,
        sep_frame: int,
        carrier_bboxes: Sequence[Tuple[float, float, float, float]],
    ) -> ClipSummary:
        fps = self.carrier_fps
        height, width = self.carrier_frames[0].shape[:2]
        vx_min, vx_max = self.sep_vx_range
        vy_min, vy_max = self.sep_vy_range
        gravity = self.gravity
        parasites = [Parasite(0.0, 0.0, 0, 0) for _ in range(self.num_parasites)]
        for parasite in parasites:
            if self.sprite_px_min > 0 or self.sprite_px_max > 0:
                parasite.abs_width = self._sample_absolute_width(width)
            else:
                parasite.scale = self.rng.uniform(self.args.sprite_scale_min, self.args.sprite_scale_max)
        out_frames: List[np.ndarray] = []
        labels: List[str] = []
        debug_rows: List[List[float]] = []
        debug_frames: List[np.ndarray] = []
        event_frame_global: Optional[int] = None
        total_frames = min(len(self.carrier_frames), len(carrier_bboxes))
        debug_save_indices: Set[int] = self._debug_frame_indices(sep_frame, total_frames) if self.debug else set()

        for idx, (frame, carrier_box) in enumerate(zip(self.carrier_frames, carrier_bboxes)):
            frame_rs = frame.copy()
            cx, cy, cw, ch = carrier_box

            # Maintain sprite geometry for each parasite based on target width.
            sprite_sizes: List[Tuple[int, int]] = []
            for parasite in parasites:
                if not parasite.detached:
                    target_width = self._determine_sprite_width(parasite, cw, width)
                    needs_update = parasite.sprite is None or parasite.last_width_px != target_width
                    if needs_update:
                        resized = self._resize_sprite(self.sprite_rgba, target_width)
                        parasite.sprite = resized
                        parasite.w = resized.shape[1]
                        parasite.h = resized.shape[0]
                        parasite.last_width_px = parasite.w
                        if parasite.first_size is None:
                            parasite.first_size = (parasite.w, parasite.h)
                elif parasite.sprite is None:
                    fallback_width = parasite.abs_width or 12
                    fallback_width = int(np.clip(fallback_width, 8, max(8, width - 4)))
                    resized = self._resize_sprite(self.sprite_rgba, fallback_width)
                    parasite.sprite = resized
                    parasite.w = resized.shape[1]
                    parasite.h = resized.shape[0]
                    parasite.last_width_px = parasite.w
                sprite_sizes.append((max(1, parasite.w), max(1, parasite.h)))

            anchors = self._anchor_positions(carrier_box, sprite_sizes)
            jitter = self.pre_sep_jitter_px

            if idx <= sep_frame:
                # Anchor sprites to the aircraft prior to separation; optional jitter applies in pixel space.
                for i, parasite in enumerate(parasites):
                    if parasite.detached:
                        continue
                    anchor_cx, anchor_cy = anchors[i]
                    jitter_x = self.rng.randint(-jitter, jitter) if jitter > 0 else 0
                    jitter_y = self.rng.randint(-jitter, jitter) if jitter > 0 else 0
                    center_x = anchor_cx + jitter_x
                    center_y = anchor_cy + jitter_y
                    parasite.anchor_center = (center_x, center_y)
                    parasite.x = float(center_x - parasite.w / 2.0)
                    parasite.y = float(center_y - parasite.h / 2.0)
                    parasite.x = float(np.clip(parasite.x, -parasite.w + 1, width - 1))
                    parasite.y = float(np.clip(parasite.y, -parasite.h + 1, height - 1))
            else:
                for parasite in parasites:
                    parasite.anchor_center = (
                        parasite.x + parasite.w / 2.0,
                        parasite.y + parasite.h / 2.0,
                    )

            if idx == sep_frame:
                for parasite in parasites:
                    if not parasite.detached:
                        parasite.detached = True
                        parasite.detach_frame = idx
                        parasite.vx = self.rng.uniform(vx_min, vx_max)
                        parasite.vy = self.rng.uniform(vy_min, vy_max)

            if idx > sep_frame:
                for parasite in parasites:
                    if parasite.detached and parasite.active:
                        parasite.x += parasite.vx
                        parasite.y += parasite.vy
                        parasite.vy += gravity
                        parasite.anchor_center = (parasite.x + parasite.w / 2.0, parasite.y + parasite.h / 2.0)
                        if (
                            parasite.x > width
                            or parasite.y > height
                            or (parasite.x + parasite.w) < 0
                            or (parasite.y + parasite.h) < 0
                        ):
                            parasite.active = False
                            parasite.k = 0

            # --- EVENT LOGIC: distance OR vertical-below-belly ---
            dist_thresh = self.event_dist_ratio * cw

            # 비행기 '배' 기준선: 중심보다 약간 아래(날개/랜딩기어 공간 감안)
            belly_y = cy + 0.15 * ch            # 필요하면 0.10~0.25 사이로 조절
            margin_px = max(8, int(0.05 * cw))  # 너무 촘촘한 판정 방지용 여유

            centers: List[Optional[Tuple[float, float]]] = []
            distance_values: List[Optional[float]] = []
            sprite_bboxes: List[Tuple[int, int, int, int, bool]] = []

            for parasite in parasites:
                if parasite.sprite is None:
                    sprite_bboxes.append((0, 0, 0, 0, False))
                    centers.append(None)
                    distance_values.append(None)
                    continue

                ix = int(round(parasite.x))
                iy = int(round(parasite.y))
                is_active = parasite.active
                sprite_bboxes.append((ix, iy, parasite.w, parasite.h, is_active))

                if is_active:
                    frame_rs = self._composite_frame(frame_rs, parasite.sprite, ix, iy)
                    center = (parasite.x + parasite.w / 2.0, parasite.y + parasite.h / 2.0)
                    centers.append(center)

                    # 기존 거리 조건
                    distance = math.hypot(center[0] - cx, center[1] - cy)
                    distance_values.append(distance)
                    cond_far = (distance > dist_thresh)

                    # 새 수직 조건: 분리됐고, '배’ 기준선보다 margin만큼 충분히 아래로 내려갔는가?
                    cond_below = (parasite.detached and (center[1] > (belly_y + margin_px)))

                    # 두 조건 중 하나라도 만족하면 카운트 증가
                    if cond_far or cond_below:
                        parasite.k += 1
                    else:
                        parasite.k = 0

                    if parasite.k >= self.event_min_frames and parasite.first_event is None:
                        parasite.first_event = idx

                    if parasite.first_event is not None:
                        if event_frame_global is None or parasite.first_event < event_frame_global:
                            event_frame_global = parasite.first_event
                else:
                    centers.append(None)
                    distance_values.append(None)
                    parasite.k = 0


            if self.args.shake:
                frame_rs = self._apply_shake(frame_rs)
            if self.args.blur and abs(idx - sep_frame) <= 2:
                base_vx = parasites[0].vx if parasites else 0.0
                frame_rs = self._apply_motion_blur(frame_rs, base_vx)
            out_frames.append(frame_rs)

            if idx < 3:
                anchor_log = [(round(a[0], 2), round(a[1], 2)) for a in anchors]
                sprite_log = [(round(p.x, 2), round(p.y, 2), p.w, p.h) for p in parasites]
                print(
                    f"[diag] frame={idx} bbox=({cx:.1f},{cy:.1f},{cw:.1f},{ch:.1f}) "
                    f"anchors={anchor_log} sprites={sprite_log}"
                )

            event_flag = 1 if (event_frame_global is not None and idx >= event_frame_global) else 0

            row = [float(idx), cx, cy, cw, ch]
            for parasite in parasites:
                row.extend([
                    float(parasite.x),
                    float(parasite.y),
                    float(parasite.w),
                    float(parasite.h),
                ])
            while len(row) < 13:
                row.extend([-1.0, -1.0, -1.0, -1.0])
            labels.append(",".join(f"{val:.2f}" for val in row) + f",{event_flag}")

            if self.debug:
                dist0 = distance_values[0] if len(distance_values) > 0 and distance_values[0] is not None else -1.0
                dist1 = distance_values[1] if len(distance_values) > 1 and distance_values[1] is not None else -1.0
                k0 = parasites[0].k if len(parasites) > 0 else -1
                k1 = parasites[1].k if len(parasites) > 1 else -1
                att0 = int(len(parasites) > 0 and not parasites[0].detached)
                att1 = int(len(parasites) > 1 and not parasites[1].detached) if len(parasites) > 1 else 0
                debug_rows.append([
                    idx,
                    cw,
                    dist0,
                    dist1,
                    dist_thresh,
                    k0,
                    k1,
                    att0,
                    att1,
                ])
                overlay = self._build_debug_overlay(frame_rs, carrier_box, anchors, sprite_bboxes, idx, sep_frame)
                debug_frames.append(overlay)
                if idx in debug_save_indices:
                    debug_path = self.debug_dir / f"{clip_uid}_frame_{idx:04d}.png"
                    cv2.imwrite(str(debug_path), overlay)

        if event_frame_global is None:
            print(f"❌ {clip_uid} failed to trigger event", file=sys.stderr)
            event_frame_value = -1
        else:
            event_frame_value = event_frame_global

        # 항상 labels/mp4/json을 쓰도록 이동 ↓
        labels_path = self.labels_dir / f"{clip_uid}.csv"
        self._write_labels(labels_path, labels)
        manifest_path = self.out_dir / f"{clip_uid}.json"
        self._write_manifest(manifest_path, clip_uid, fps, width, height, sep_frame)
        out_path = self.out_dir / f"{clip_uid}.mp4"
        writer_kind = self._write_video(out_path, out_frames, fps)
        if self.frames_per_clip > 0:
            self._export_frames(clip_uid, out_frames, labels)

        first_size = next((p.first_size for p in parasites if p.first_size is not None), (0, 0))
        first_size = (int(first_size[0]), int(first_size[1])) if first_size else (0, 0)

        if self.debug:
            dbg_csv = self.debug_dir / f"dbg_clip_{clip_uid}.csv"
            with dbg_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["frame", "cw", "dist0", "dist1", "thresh", "k0", "k1", "att0", "att1"])
                writer.writerows(debug_rows)
            if debug_frames:
                self._write_debug_video(self.debug_dir / f"dbg_clip_{clip_uid}.mp4", debug_frames, fps)


        return ClipSummary(
            uid=clip_uid,
            sep_frame=sep_frame,
            event_frame=event_frame_value,
            fps=fps,
            width=width,
            height=height,
            sprite_px=first_size,
            parasites=self.num_parasites,
            anchor_mode=self.anchor_mode,
            writer_kind=writer_kind,
            out_path=out_path,
        )

    def _resize_sprite(self, sprite: np.ndarray, desired_width: int) -> np.ndarray:
        base_h, base_w = sprite.shape[:2]
        desired_width = max(1, int(round(desired_width)))
        scale = desired_width / float(base_w)
        desired_height = max(2, int(round(base_h * scale)))
        # (리버트) 간단히 AREA 리사이즈
        return cv2.resize(sprite, (desired_width, desired_height), interpolation=cv2.INTER_AREA)

    def _apply_unsharp(self, img_bgr: np.ndarray, amount: float) -> np.ndarray:
        radius = max(1, int(round(3 * amount)))
        if radius % 2 == 0:
            radius += 1
        blur = cv2.GaussianBlur(img_bgr, (radius, radius), 0)
        img_f = img_bgr.astype(np.float32)
        blur_f = blur.astype(np.float32)
        sharpened = (1.0 + amount) * img_f - amount * blur_f
        return np.clip(sharpened, 0.0, 255.0).astype(np.uint8)

    def _sample_absolute_width(self, frame_width: int) -> int:
        min_px = self.sprite_px_min if self.sprite_px_min > 0 else 0
        max_px = self.sprite_px_max if self.sprite_px_max > 0 else 0
        if min_px == 0 and max_px == 0:
            raise ValueError("Absolute sprite width requested without bounds")
        if min_px == 0:
            min_px = max_px
        if max_px == 0:
            max_px = min_px
        max_frame = max(8, frame_width - 4)
        min_px = int(np.clip(min_px, 8, max_frame))
        max_px = int(np.clip(max_px, 8, max_frame))
        if max_px < min_px:
            min_px, max_px = max_px, min_px
        if min_px == max_px:
            return min_px
        return self.rng.randint(min_px, max_px)

    def _determine_sprite_width(self, parasite: Parasite, carrier_w: float, frame_w: int) -> int:
        max_frame = max(8, frame_w - 4)
        if parasite.abs_width is not None:
            return int(np.clip(parasite.abs_width, 8, max_frame))
        scale = parasite.scale
        if scale is None:
            scale = self.rng.uniform(self.args.sprite_scale_min, self.args.sprite_scale_max)
            parasite.scale = scale
        target = int(round(carrier_w * scale))
        target = max(8, target)
        target = min(target, max_frame)
        return max(8, target)

    def _debug_frame_indices(self, sep_frame: int, total_frames: int) -> Set[int]:
        base = {0, 30, 60, sep_frame - 2, sep_frame, sep_frame + 2}
        return {idx for idx in base if 0 <= idx < total_frames}

    def _build_debug_overlay(
        self,
        frame: np.ndarray,
        carrier_box: Tuple[float, float, float, float],
        anchors: Sequence[Tuple[float, float]],
        sprite_bboxes: Sequence[Tuple[int, int, int, int, bool]],
        frame_idx: int,
        sep_frame: int,
    ) -> np.ndarray:
        overlay = frame.copy()
        cx, cy, cw, ch = carrier_box
        x0 = int(round(cx - cw / 2.0))
        y0 = int(round(cy - ch / 2.0))
        x1 = int(round(cx + cw / 2.0))
        y1 = int(round(cy + ch / 2.0))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
        for ax, ay in anchors:
            cv2.circle(overlay, (int(round(ax)), int(round(ay))), 3, (0, 0, 255), -1)
        for sx, sy, sw, sh, active in sprite_bboxes:
            if sw <= 0 or sh <= 0:
                continue
            color = (255, 255, 0) if active else (128, 128, 128)
            cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), color, 2)
        if frame_idx == sep_frame:
            x_line = overlay.shape[1] // 2
            cv2.line(overlay, (x_line, 0), (x_line, overlay.shape[0] - 1), (0, 255, 255), 2)
        cv2.putText(
            overlay,
            f"f={frame_idx}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _match_color(self, sprite_rgba: np.ndarray) -> np.ndarray:
        alpha = sprite_rgba[:, :, 3:4]
        if np.count_nonzero(alpha) == 0:
            return sprite_rgba
        sprite_mean = (sprite_rgba[:, :, :3] * alpha).sum(axis=(0, 1)) / alpha.sum()
        bg_frame = self.carrier_frames[0].astype(np.float32) / 255.0
        bg_mean = bg_frame.mean(axis=(0, 1), keepdims=True)
        scale = np.clip(bg_mean / (sprite_mean + 1e-6), 0.5, 1.5)
        matched = sprite_rgba.copy()
        matched[:, :, :3] = np.clip(sprite_rgba[:, :, :3] * scale, 0.0, 1.0)
        return matched

    def _anchor_positions(
        self,
        carrier_box: Tuple[float, float, float, float],
        sprite_sizes: Sequence[Tuple[int, int]],
    ) -> List[Tuple[float, float]]:
        """기본 앵커(센터/날개)만 계산. 오프셋/지터 없음, 스프라이트 중심 좌표 반환."""
        cx, cy, cw, ch = carrier_box
        anchors: List[Tuple[float, float]] = []
        mode = self.anchor_mode if self.anchor_mode in ANCHOR_MODES else "wings"

        if mode == "wings":
            left_center, right_center = wing_anchors(cx, cy, cw, ch, self.wing_dx_ratio, self.wing_dy_ratio)
            centers = [left_center]
            if self.num_parasites > 1:
                centers.append(right_center)
            for i in range(self.num_parasites):
                base_x, base_y = centers[min(i, len(centers) - 1)]
                anchors.append((base_x, base_y))
            return anchors

        # top_center / center
        for i in range(self.num_parasites):
            sprite_w, sprite_h = sprite_sizes[i]
            offset = (i - (self.num_parasites - 1) / 2.0) * sprite_w * 1.2
            if mode == "top_center":
                center_y = cy - ch / 2.0 - sprite_h / 2.0 - 5.0
            else:  # center
                center_y = cy
            center_x = cx + offset
            anchors.append((center_x, center_y))
        return anchors


    def _composite_frame(self, frame_bgr: np.ndarray, sprite_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
        sprite_h, sprite_w = sprite_rgba.shape[:2]
        h, w = frame_bgr.shape[:2]
        if x >= w or y >= h or x + sprite_w <= 0 or y + sprite_h <= 0:
            return frame_bgr
        if self.shadow_strength > 0:
            frame_float = frame_bgr.astype(np.float32) / 255.0
            frame_float = self._apply_shadow(frame_float, x + sprite_w / 2, y + sprite_h, sprite_w, sprite_h)
            frame_bgr = np.clip(frame_float * 255.0, 0, 255).astype(np.uint8)
        safe_composite_rgba_bgr(frame_bgr, sprite_rgba, x, y)
        return frame_bgr

    def _apply_shadow(self, frame: np.ndarray, x_center: float, y_base: float, sprite_w: int, sprite_h: int) -> np.ndarray:
        shadow = np.zeros(frame.shape[:2], dtype=np.float32)
        center = (int(round(x_center)), int(round(y_base + sprite_h * 0.1)))
        axes = (max(1, int(sprite_w * 0.6)), max(1, int(sprite_h * 0.25)))
        cv2.ellipse(shadow, center, axes, 0, 0, 360, self.shadow_strength, -1)
        shadow = cv2.GaussianBlur(shadow, (31, 31), 0)
        frame[:, :, :] = np.clip(frame - shadow[:, :, None], 0.0, 1.0)
        return frame

    def _apply_shake(self, frame: np.ndarray) -> np.ndarray:
        tx = self.rng.uniform(-1.0, 1.0)
        ty = self.rng.uniform(-1.0, 1.0)
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT101)

    def _apply_motion_blur(self, frame: np.ndarray, vx: float) -> np.ndarray:
        kernel_size = 5
        direction = int(np.sign(vx)) or 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if direction >= 0:
            for i in range(kernel_size):
                kernel[kernel_size // 2, i] = 1.0
        else:
            for i in range(kernel_size):
                kernel[i, kernel_size // 2] = 1.0
        kernel /= kernel.sum()
        return cv2.filter2D(frame, -1, kernel)

    def _write_labels(self, path: Path, lines: Sequence[str]) -> None:
        header = "frame,carrier_cx,carrier_cy,carrier_w,carrier_h,para1_x,para1_y,para1_w,para1_h,para2_x,para2_y,para2_w,para2_h,event\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            for line in lines:
                fh.write(line + "\n")

    def _write_manifest(self, path: Path, uid: str, fps: int, width: int, height: int, sep_frame: int) -> None:
        payload = {
            "uid": uid,
            "fps": fps,
            "width": width,
            "height": height,
            "sep_frame": sep_frame,
            "geo": {
                "lat": 52.5 + 0.01 * int(uid.split("_")[-1]),
                "lon": 13.3 + 0.01 * int(uid.split("_")[-1]),
                "dpp_lat": 1.0e-5,
                "dpp_lon": 1.0e-5,
            },
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _write_video(self, out_path: Path, frames: Sequence[np.ndarray], fps: int) -> str:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if self.args.use_ffmpeg or not writer.isOpened():
            if writer.isOpened():
                writer.release()
            return self._write_video_ffmpeg(out_path, frames, fps)
        for frame in frames:
            writer.write(frame)
        writer.release()
        return "cv2"

    def _write_video_ffmpeg(self, out_path: Path, frames: Sequence[np.ndarray], fps: int) -> str:
        tmp_dir = Path(tempfile.mkdtemp(prefix="synth_frames_"))
        try:
            for idx, frame in enumerate(frames):
                cv2.imwrite(str(tmp_dir / f"frame_{idx:05d}.png"), frame)
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_dir / "frame_%05d.png"),
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', 'ignore')}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return "ffmpeg"

    def _write_debug_video(self, out_path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
        if not frames:
            return
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open debug writer for {out_path}")
        for frame in frames:
            writer.write(frame)
        writer.release()

    def _export_frames(self, clip_uid: str, frames: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        sample_dir = self.out_dir / f"{clip_uid}_frames"
        sample_dir.mkdir(parents=True, exist_ok=True)
        total = len(frames)
        step = max(total // self.frames_per_clip, 1)
        for idx in range(0, total, step):
            cv2.imwrite(str(sample_dir / f"frame_{idx:05d}.jpg"), frames[idx])
        labels_path = sample_dir / "labels.csv"
        self._write_labels(labels_path, labels)

def _track_csrt(self, frames: Sequence[np.ndarray],
                init_xywh: Tuple[int,int,int,int]) -> List[Tuple[float,float,float,float]]:
    # returns list of (cx,cy,w,h) per frame
    tracker = cv2.legacy.TrackerCSRT_create()
    x,y,w,h = init_xywh
    first = frames[0]
    ok = tracker.init(first, (float(x), float(y), float(w), float(h)))
    if not ok:
        raise RuntimeError("CSRT init failed")
    out = []
    for f in frames:
        ok, box = tracker.update(f)
        if not ok:
            # tracking lost → keep last or fallback to center box
            if out:
                cx,cy,w,h = out[-1]
                out.append((cx,cy,w,h,h))  # small bug guard; fixed below
                out[-1]=(out[-1][0], out[-1][1], out[-1][2], out[-1][3])  # no-op
                continue
            H,W = f.shape[:2]
            cw,ch = int(W*0.4), int(H*0.2)
            out.append((W/2, H/2, cw, ch))
            continue
        x,y,w,h = box
        cx, cy = x + w/2.0, y + h/2.0
        out.append((cx, cy, float(w), float(h)))
    return out

def _estimate_carrier_track(self, frames):
    # 1) 명시 bbox 있으면 CSRT로
    if self.args.track == "csrt" and self.args.carrier_bbox:
        X,Y,W,H = self.args.carrier_bbox
        return self._track_csrt(frames, (int(X),int(Y),int(W),int(H)))

    # 2) CSRT인데 init bbox가 없다? GUI 가능하면 selectROI, 아니면 경고 후 휴리스틱
    if self.args.track == "csrt" and not self.args.carrier_bbox:
        try:
            f0 = frames[0].copy()
            r = cv2.selectROI("Select carrier bbox", f0, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            if r and r[2] > 0 and r[3] > 0:
                return self._track_csrt(frames, (int(r[0]),int(r[1]),int(r[2]),int(r[3])))
        except Exception as e:
            print(f"[warn] ROI GUI not available ({e}); falling back to heuristic.", file=sys.stderr)
    # 3) 기본: 기존 휴리스틱 + EMA
    smoothed = []
    prev = None
    for frame in frames:
        bbox = self._estimate_carrier_bbox(frame)
        if prev is None:
            prev = bbox
        else:
            prev = tuple(prev[i]*0.8 + bbox[i]*0.2 for i in range(4))  # type: ignore
        smoothed.append(prev)  # type: ignore[arg-type]
    return smoothed

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic parasite launch clips")
    parser.add_argument("--carrier_bbox", nargs=4, type=int, default=None,
                        metavar=("X","Y","W","H"),
                        help="init bbox (pixels) for tracking; strongly recommended")
    parser.add_argument("--track", default="none", choices=["none","csrt"],
                    help="carrier tracking mode")
    parser.add_argument("--roi", action="store_true",
                        help="use GUI ROI picker on first frame (requires display)")
    parser.add_argument("--color_match", type=int, default=1, help="1=match sprite brightness to BG")
    parser.add_argument("--shadow_strength", type=float, default=0.35, help="0..1 under-sprite shadow strength")
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--sprite", required=True)
    parser.add_argument("--out_dir", default="data/synthetic")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sep_frame_min", type=int, default=80)
    parser.add_argument("--sep_frame_max", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_secs", type=int, default=15)
    parser.add_argument("--blur", type=int, default=1)
    parser.add_argument("--shake", type=int, default=1)
    parser.add_argument("--num_parasites", type=int, default=2)
    parser.add_argument("--anchor_mode", default="wings", choices=sorted(ANCHOR_MODES))
    parser.add_argument("--sprite_scale_min", type=float, default=0.04)
    parser.add_argument("--sprite_scale_max", type=float, default=0.08)
    parser.add_argument("--sprite_px_min", type=int, default=0,
                        help="absolute sprite width min (px), overrides relative scaling when >0")
    parser.add_argument("--sprite_px_max", type=int, default=0,
                        help="absolute sprite width max (px), overrides relative scaling when >0")
    parser.add_argument("--sprite_resize", default="lanczos",
                        choices=["nearest", "area", "lanczos"],
                        help="sprite resize kernel")
    parser.add_argument("--sprite_unsharp", type=float, default=0.0,
                        help="unsharp mask amount applied after resize (0 disables)")
    parser.add_argument("--sprite_premultiply", type=int, default=1,
                        help="1=premultiply RGB by alpha before compositing")
    parser.add_argument("--wing_dx_ratio", type=float, default=0.35)
    parser.add_argument("--wing_dy_ratio", type=float, default=0.25)
    parser.add_argument("--anchor_px_offset", nargs=2, type=float, default=[0.0, 0.0],
                        metavar=("OX", "OY"), help="pixel offset added to anchor centers")
    parser.add_argument("--pre_sep_jitter_px", type=int, default=0)
    parser.add_argument("--event_dist_ratio", type=float, default=1.05,
                    help="distance threshold as ratio of carrier_w (default 1.05)")
    parser.add_argument("--event_min_frames", type=int, default=2,
                        help="consecutive frames required (default 2)")
    parser.add_argument("--sep_vx_range", nargs=2, type=float, default=[-4, 4],
                        help="horizontal velocity range (min max)")
    parser.add_argument("--sep_vy_range", nargs=2, type=float, default=[-2, -1],
                        help="vertical velocity range (min max)")
    parser.add_argument("--gravity", type=float, default=0.6)
    parser.add_argument("--use_ffmpeg", type=int, default=0)
    parser.add_argument("--frames_per_clip", type=int, default=0)
    parser.add_argument("--keep_size", type=int, default=0)
    parser.add_argument("--debug", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    synth = Synthesizer(args)
    synth.generate()


if __name__ == "__main__":
    main(sys.argv[1:])
