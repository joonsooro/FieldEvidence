from pathlib import Path
import json
from src.c2.scheduler import approve_and_dispatch, AssetPool

def test_approve_and_dispatch_appends_three_states(tmp_path: Path):
    audit_path = tmp_path / "c2_audit.jsonl"

    # Mock recommendation (IMMEDIATE)
    reco = {
        "event_id": "mock#1",
        "uid": 12345,
        "risk": {"status": "IMMEDIATE"},
        "action": {"type": "INTERCEPT", "distance_m": 1200.0, "bearing_deg": 45.0, "eta_sec": 60.0},
    }
    # Minimal estimation snapshot
    estimation = {
        "est_pos": {"lat": 52.5, "lon": 13.3},
        "est_vel": {"mps": 10.0, "heading_deg": 90.0},
        "confidence": 0.9,
    }
    cfg = {"asset_speed_mps": 25.0}

    pool = AssetPool(["sim_strike_01"])
    out = approve_and_dispatch(reco, estimation, cfg, audit_path=audit_path, pool=pool, now_ns=1000)

    # Validate return summary
    assert out["asset_id"] == "sim_strike_01"
    assert out["dispatch_id"].startswith("disp_")
    assert out["eta_sec"] > 0
    assert "states" in out and len(out["states"]) == 3

    # Validate audit file written with 3 lines/states
    assert audit_path.exists()
    lines = [json.loads(s) for s in audit_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(lines) == 3
    states = [l["state"] for l in lines]
    assert states == ["ASSIGNED", "LAUNCHED", "INTERCEPT_EXPECTED"]
    # monotonic timestamps
    ts = [l["ts_ns"] for l in lines]
    assert ts[0] == 1000
    assert ts[1] > ts[0]
    assert ts[2] > ts[1]

