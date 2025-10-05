from src.c2.prioritizer import compute_risk


def _cfg():
    return {
        "asset_speed_mps": 12.0,
        "weights": {"intent": 0.8, "proximity": 10.0, "recent": 0.6},
        "thresholds": {"IMMEDIATE": 1.6, "SUSPECT": 1.1},
    }


def test_risk_immediate_close_fast():
    cfg = _cfg()
    target_lat, target_lon = 52.5200, 13.4050
    est = {
        "est_pos": {"lat": target_lat, "lon": target_lon},  # on top of target
        "est_vel": {"mps": 15.0, "heading_deg": 90.0},
        "confidence": 1.0,
    }
    r = compute_risk(est, target_lat, target_lon, cfg)
    assert isinstance(r, dict)
    assert r["status"] == "IMMEDIATE"
    assert r["score"] >= cfg["thresholds"]["IMMEDIATE"]


def test_risk_not_immediate_far_slow_low_conf():
    cfg = _cfg()
    target_lat, target_lon = 52.5200, 13.4050
    est = {
        "est_pos": {"lat": 52.0, "lon": 14.0},  # far away
        "est_vel": {"mps": 2.0, "heading_deg": 0.0},
        "confidence": 0.1,
    }
    r = compute_risk(est, target_lat, target_lon, cfg)
    assert r["status"] in {"SUSPECT", "INFO"}
    assert r["score"] < cfg["thresholds"]["IMMEDIATE"]

