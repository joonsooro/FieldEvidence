import json
from pathlib import Path

from src.c2.recommendation import make_recommendation


SCHEMA_PATH = Path("dev/RECOMM_SCHEMA.json")


def _cfg():
    return {
        "asset_speed_mps": 12.0,
        "weights": {"intent": 0.8, "proximity": 10.0, "recent": 0.6},
        "thresholds": {"IMMEDIATE": 1.6, "SUSPECT": 1.1},
    }


def _load_schema() -> dict:
    assert SCHEMA_PATH.exists(), "Schema file missing"
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validate_against_schema(obj: dict, schema: dict) -> None:
    # Minimal manual validation to avoid extra deps
    req = schema.get("required", [])
    for k in req:
        assert k in obj, f"missing required key: {k}"
    # Spot check types for a subset
    assert isinstance(obj["event_id"], str)
    assert isinstance(obj["ts_ns"], int)
    assert isinstance(obj["risk"], dict)
    assert isinstance(obj["action"], dict)
    assert isinstance(obj["message"], str)
    assert isinstance(obj["created_ns"], int)
    # Risk details
    r = obj["risk"]
    assert r["status"] in {"IMMEDIATE", "SUSPECT", "INFO"}
    assert isinstance(r["score"], (int, float))
    # Action type
    assert obj["action"]["type"] == "INTERCEPT"


def test_make_recommendation_immediate_valid_schema():
    schema = _load_schema()
    cfg = _cfg()
    target_lat, target_lon = 52.5200, 13.4050
    event = {"event_id": "42#1000", "uid": 42, "ts_ns": 1234567890}
    est = {
        "est_pos": {"lat": target_lat, "lon": target_lon},
        "est_vel": {"mps": 15.0, "heading_deg": 90.0},
        "confidence": 1.0,
    }

    rec = make_recommendation(est, event, target_lat, target_lon, cfg)
    assert rec is not None, "Expected recommendation for IMMEDIATE risk"
    _validate_against_schema(rec, schema)


def test_make_recommendation_non_immediate_returns_none():
    cfg = _cfg()
    target_lat, target_lon = 52.5200, 13.4050
    event = {"event_id": "43#2000", "uid": 43, "ts_ns": 987654321}
    est = {
        "est_pos": {"lat": 52.0, "lon": 14.0},
        "est_vel": {"mps": 2.0, "heading_deg": 0.0},
        "confidence": 0.2,
    }

    rec = make_recommendation(est, event, target_lat, target_lon, cfg)
    assert rec is None

