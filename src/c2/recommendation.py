from __future__ import annotations

import time
from typing import Dict, Optional

from .prioritizer import compute_risk, haversine


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brng


def make_recommendation(
    estimation: Dict,
    event: Dict,
    target_lat: float,
    target_lon: float,
    cfg: Dict,
) -> Optional[Dict]:
    """Generate an actionable recommendation. By default only IMMEDIATE triggers,
    but this can be relaxed via cfg['policy']['recommend_min_status'].

    Args:
        estimation: Tracker output dict with keys `est_pos`, `est_vel`, `confidence`.
        event: Original normalized event with at least `event_id`, `uid`, `ts_ns`.
        target_lat, target_lon: Defended asset location.
        cfg: Configuration dict with weights/thresholds and asset_speed_mps and optional:
             cfg['policy'] = {
                 'recommend_min_status': 'IMMEDIATE' | 'SUSPECT' | 'INFO',
                 'suspect_monitor_window_sec': float (default 30.0)
             }

    Returns:
        Recommendation dict if policy permits for the given status; otherwise None.
    """
    risk = compute_risk(estimation, target_lat, target_lon, cfg)

    # Policy gate: which minimum status should emit a recommendation?
    policy = dict(cfg.get("policy", {}))
    min_status = str(policy.get("recommend_min_status", "IMMEDIATE")).upper()
    STATUS_RANK = {"INFO": 0, "SUSPECT": 1, "IMMEDIATE": 2}
    cur_status = str(risk.get("status", "INFO")).upper()
    if STATUS_RANK.get(cur_status, 0) < STATUS_RANK.get(min_status, 2):
        return None

    # Common fields
    lat = float(estimation["est_pos"]["lat"])
    lon = float(estimation["est_pos"]["lon"])
    mps = float(estimation["est_vel"].get("mps", 0.0))
    distance_m = float(risk["factors"]["distance_m"]) if "factors" in risk else haversine(lat, lon, target_lat, target_lon)
    bearing = _bearing_deg(lat, lon, float(target_lat), float(target_lon))

    # Compute ETA only for moving targets
    eta_sec: Optional[float] = None
    if mps > 0.1:
        eta_sec = distance_m / mps

    # Branch by severity
    if cur_status == "IMMEDIATE":
        action = {
            "type": "INTERCEPT",
            "distance_m": distance_m,
            "bearing_deg": bearing,
            "eta_sec": eta_sec,
        }
        msg = (
            f"IMMEDIATE: uid={event.get('uid')} {distance_m:.0f} m from asset, "
            f"bearing {bearing:.0f}°. Speed {mps:.1f} m/s."
        )
    elif cur_status == "SUSPECT":
        monitor_sec = float(policy.get("suspect_monitor_window_sec", 30.0))
        action = {
            "type": "OBSERVE",  # softer action for SUSPECT
            "distance_m": distance_m,
            "bearing_deg": bearing,
            "monitor_window_sec": monitor_sec,
            "eta_sec": eta_sec,  # informative only
        }
        msg = (
            f"SUSPECT: uid={event.get('uid')} {distance_m:.0f} m, "
            f"bearing {bearing:.0f}°. Speed {mps:.1f} m/s. "
            f"Recommend observe/hold for ~{monitor_sec:.0f}s; auto-promote if score rises."
        )
    else:
        # INFO (or unknown) -> do not emit recommendation unless policy lowered to INFO
        action = {
            "type": "LOG",
            "distance_m": distance_m,
            "bearing_deg": bearing,
            "eta_sec": eta_sec,
        }
        msg = (
            f"INFO: uid={event.get('uid')} {distance_m:.0f} m, "
            f"bearing {bearing:.0f}°. Speed {mps:.1f} m/s."
        )

    rec = {
        "event_id": str(event.get("event_id", "")),
        "uid": event.get("uid"),
        "ts_ns": int(event.get("ts_ns", 0)),
        "severity": cur_status,  # <-- surface status for UI logic
        "risk": risk,
        "action": action,
        "message": msg,
        "created_ns": time.time_ns(),
    }
    return rec


__all__ = ["make_recommendation"]

