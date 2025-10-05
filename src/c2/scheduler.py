from __future__ import annotations

import time
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default audit sink
AUDIT_PATH = Path("out/c2_audit.jsonl")

# --- Small equirectangular projection helpers (good enough for short distances) ---
# NOTE: This is a simple, fast approximation; fine for a demo.
_EARTH_R = 6371000.0  # meters

def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def _rad2deg(r: float) -> float:
    return r * 180.0 / math.pi

def project_point(lat: float, lon: float, heading_deg: float, dist_m: float) -> Tuple[float, float]:
    """
    Project (lat, lon) by moving 'dist_m' meters along 'heading_deg' (north=0°, east=90°).
    Equirectangular approximation centered at the starting latitude.
    """
    lat_r = _deg2rad(lat)
    lon_r = _deg2rad(lon)
    brg = _deg2rad(heading_deg)

    # meters → degrees with local scale
    dlat = (dist_m * math.cos(brg)) / _EARTH_R
    dlon = (dist_m * math.sin(brg)) / (_EARTH_R * math.cos(lat_r))

    return (_rad2deg(lat_r + dlat), _rad2deg(lon_r + dlon))


@dataclass
class AssetPool:
    """Very small in-memory pool of asset IDs."""
    assets: List[str]
    busy: set[str] = None

    def __post_init__(self):
        if self.busy is None:
            self.busy = set()

    def acquire(self) -> Optional[str]:
        for a in self.assets:
            if a not in self.busy:
                self.busy.add(a)
                return a
        return None

    def release(self, asset_id: str) -> None:
        self.busy.discard(asset_id)


def _now_ns() -> int:
    return time.time_ns()


def _append_audit(rec: Dict, audit_path: Path = AUDIT_PATH) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def approve_and_dispatch(
    reco: Dict,
    estimation: Dict,
    cfg: Dict,
    audit_path: Path = AUDIT_PATH,
    pool: Optional[AssetPool] = None,
    now_ns: Optional[int] = None,
) -> Dict:
    """
    Turn an approved recommendation (IMMEDIATE) into a simulated dispatch timeline.

    Args:
      reco: recommendation dict from src/c2/recommendation.make_recommendation()
            expected keys: "event_id", "uid", "risk", "action":{type,distance_m,bearing_deg,eta_sec?}
      estimation: estimator output dict with "est_pos":{"lat","lon"}, "est_vel":{"mps","heading_deg"}
      cfg: config with "asset_speed_mps" at least
      audit_path: JSONL sink for timeline
      pool: optional AssetPool; default pool with ["sim_strike_01","sim_strike_02"]
      now_ns: optional fixed clock for test determinism

    Returns:
      Summary dict with dispatch_id, asset_id, eta_sec, intercept position.
    """
    if reco.get("action", {}).get("type") != "INTERCEPT":
        raise ValueError("approve_and_dispatch requires an INTERCEPT recommendation")

    if pool is None:
        pool = AssetPool(["sim_strike_01", "sim_strike_02"])

    asset_id = pool.acquire()
    if asset_id is None:
        raise RuntimeError("No asset available")

    # Pull inputs
    tgt_uid = reco.get("uid")
    event_id = reco.get("event_id")
    dist_m = float(reco.get("action", {}).get("distance_m", 0.0))
    eta_sec = reco.get("action", {}).get("eta_sec")  # may be None; we'll recompute
    asset_speed = float(cfg.get("asset_speed_mps", 25.0))

    # Recompute ETA from asset speed if not provided
    if not eta_sec:
        # If the asset flies at asset_speed toward the current target position:
        # we use the same ETA definition as "distance / asset_speed".
        eta_sec = dist_m / max(0.1, asset_speed)

    # Intercept point: project target forward along its *own* heading by ETA
    lat = float(estimation["est_pos"]["lat"])
    lon = float(estimation["est_pos"]["lon"])
    heading = float(estimation["est_vel"].get("heading_deg", 0.0))
    speed_mps = float(estimation["est_vel"].get("mps", 0.0))
    # Predict how far target will travel by ETA (closing speed ignored for simplicity)
    proj_dist = speed_mps * float(eta_sec)
    intercept_lat, intercept_lon = project_point(lat, lon, heading, proj_dist)

    # Timeline stamps
    t0 = now_ns if now_ns is not None else _now_ns()
    t_launch = t0 + int(0.5 * 1e9)  # +0.5s
    t_intercept = t0 + int(float(eta_sec) * 1e9)

    # Build a deterministic dispatch_id
    dispatch_id = f"disp_{t0}"

    # Common context
    base_ctx = {
        "dispatch_id": dispatch_id,
        "asset_id": asset_id,
        "target_uid": tgt_uid,
        "event_id": event_id,
        "eta_sec": float(eta_sec),
        "intercept": {"lat": intercept_lat, "lon": intercept_lon},
    }

    # Append three states: ASSIGNED, LAUNCHED, INTERCEPT_EXPECTED
    _append_audit({**base_ctx, "state": "ASSIGNED", "ts_ns": t0}, audit_path)
    _append_audit({**base_ctx, "state": "LAUNCHED", "ts_ns": t_launch}, audit_path)
    _append_audit({**base_ctx, "state": "INTERCEPT_EXPECTED", "ts_ns": t_intercept}, audit_path)

    # Return summary for UI / caller
    return {
        **base_ctx,
        "states": [
            {"state": "ASSIGNED", "ts_ns": t0},
            {"state": "LAUNCHED", "ts_ns": t_launch},
            {"state": "INTERCEPT_EXPECTED", "ts_ns": t_intercept},
        ],
    }


__all__ = ["AssetPool", "approve_and_dispatch", "project_point"]

