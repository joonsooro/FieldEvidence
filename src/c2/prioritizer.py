import math
from typing import Dict


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (meters)."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (degrees 0–360) from (lat1,lon1) to (lat2,lon2)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings in degrees (0–180)."""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_risk(estimation: Dict, target_lat: float, target_lon: float, cfg: Dict) -> Dict:
    """
    Compute risk score and classification using TTI(시간-접촉), 방위 정렬, 지오펜스 보너스.

    estimation: {
      "est_pos": {"lat": float, "lon": float},
      "est_vel": {"mps": float, "heading_deg": float|None},
      "confidence": float,            # 시간 감쇠 포함 (0..1)
      "intent_score"?: float          # (선택) 행동 기반 의도 점수 (0..1)
    }

    cfg keys (with sensible defaults if missing):
      weights: {intent, proximity, recent, geo_bonus}
      thresholds: {IMMEDIATE_score, SUSPECT_score} (fallback to {IMMEDIATE, SUSPECT})
      immediate_rules: {tti_sec_hard, tti_sec_geofence, intent_min, conf_min, tti_ref_sec}
      assets: {protected_lat, protected_lon, geofence_radius_m}
      asset_speed_mps: float
    """
    # --- config & defaults ---
    w = cfg.get("weights", {})
    alpha = float(w.get("intent", 0.45))
    beta = float(w.get("proximity", 0.35))
    gamma = float(w.get("recent", 0.20))
    delta = float(w.get("geo_bonus", 0.0))

    thr = cfg.get("thresholds", {})
    thr_immediate_score = float(thr.get("IMMEDIATE_score", thr.get("IMMEDIATE", 0.8)))
    thr_suspect_score = float(thr.get("SUSPECT_score", thr.get("SUSPECT", 0.5)))

    ir = cfg.get("immediate_rules", {})
    tti_ref_sec = float(ir.get("tti_ref_sec", 20.0))
    tti_sec_hard = float(ir.get("tti_sec_hard", 12.0))
    tti_sec_geofence = float(ir.get("tti_sec_geofence", 18.0))
    intent_min = float(ir.get("intent_min", 0.75))
    conf_min = float(ir.get("conf_min", 0.60))

    assets = cfg.get("assets", {})
    prot_lat = float(assets.get("protected_lat", target_lat))
    prot_lon = float(assets.get("protected_lon", target_lon))
    geofence_radius_m = float(assets.get("geofence_radius_m", 75.0))

    asset_speed_mps = float(cfg.get("asset_speed_mps", 12.0))

    # --- inputs ---
    plat = float(estimation["est_pos"]["lat"])  # platform(lat)
    plon = float(estimation["est_pos"]["lon"])  # platform(lon)
    mps = float(estimation.get("est_vel", {}).get("mps", 0.0))
    heading = estimation.get("est_vel", {}).get("heading_deg", None)
    confidence = float(estimation.get("confidence", 0.0))

    # Optional behavior intent override
    intent_score = estimation.get("intent_score", None)
    if intent_score is None:
        # Fallback: 속도 임계 기반 (단, 0..1 범위로 부드럽게)
        # 0.5 @ 0 mps  → 1.0 @ >= asset_speed_mps
        intent_score = 0.5 + 0.5 * _clamp(mps / max(1e-6, asset_speed_mps), 0.0, 1.0)
    intent_score = float(intent_score)

    # --- geometry: distance & alignment ---
    # 보호 자산까지의 거리 (또는 함수 인자로 받은 target 좌표)
    dist_m = haversine(plat, plon, target_lat, target_lon)

    # 진행 방위 vs 목표(자산) 방위 차이
    bearing_to_asset = bearing_deg(plat, plon, target_lat, target_lon)
    if heading is None:
        # 방위 모르면 접근 가정(보수적). 필요시 0.5로 완화 가능.
        approach = 1.0
    else:
        ddeg = _angle_diff_deg(float(heading), float(bearing_to_asset))
        approach = max(0.0, math.cos(math.radians(ddeg)))  # 1=정면 접근, 0=직각, <0=이탈(0 처리)

    closing_speed = mps * approach  # m/s (접근 성분)
    if closing_speed <= 1e-6:
        tti_sec = float("inf")
    else:
        tti_sec = dist_m / closing_speed

    # --- proximity term: TTI 정규화 (작을수록 1에 수렴) ---
    proximity_term = tti_ref_sec / max(tti_ref_sec, tti_sec)

    # --- geo bonus ---
    dist_to_protected = haversine(plat, plon, prot_lat, prot_lon)
    geo_inside = dist_to_protected <= geofence_radius_m
    geo_bonus = delta if geo_inside else 0.0

    # --- composite score ---
    score = (
        alpha * intent_score
        + beta * proximity_term
        + gamma * confidence
        + geo_bonus
    )

    # --- multi-condition decision (A/B/C) ---
    status = "INFO"
    # A: 시간 촉박 + 의도 + 신뢰
    if (tti_sec <= tti_sec_hard) and (intent_score >= intent_min) and (confidence >= conf_min):
        status = "IMMEDIATE"
    # B: 정책 구역 내부면 완화된 임계
    elif geo_inside and (tti_sec <= tti_sec_geofence):
        status = "IMMEDIATE"
    # C: 점수 기반
    elif score >= thr_immediate_score:
        status = "IMMEDIATE"
    elif score >= thr_suspect_score:
        status = "SUSPECT"

    return {
        "score": round(float(score), 4),
        "status": status,
        "factors": {
            "distance_m": float(dist_m),
            "proximity_term": float(proximity_term),
            "tti_sec": float(tti_sec if math.isfinite(tti_sec) else 1e9),
            "approach": float(approach),
            "closing_speed_mps": float(closing_speed),
            "intent": float(intent_score),
            "recent_rate": float(confidence),
            "geo_inside": bool(geo_inside),
            "geo_bonus": float(geo_bonus),
            "bearing_to_asset_deg": float(bearing_to_asset),
            "heading_deg": float(heading) if heading is not None else None,
        },
    }


__all__ = ["haversine", "bearing_deg", "compute_risk"]
