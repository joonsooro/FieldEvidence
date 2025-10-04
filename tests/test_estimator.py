from math import cos, radians

from src.c2.estimator import Tracker


def test_velocity_heading_and_confidence_sanity():
    # Setup tracker
    tr = Tracker(window_sec=30.0, decay_tau_sec=20.0)

    # Choose a base location where lon meters/deg is stable
    lat0 = 52.0
    lon0 = 13.0

    # Move ~10 m east every second
    meters_per_deg_lon_at_lat = 111_320.0 * cos(radians(lat0))
    dlon_10m = 10.0 / meters_per_deg_lon_at_lat

    t0 = 0
    t1 = int(1e9)
    t2 = int(2e9)

    tr.update({"ts_ns": t0, "lat": lat0, "lon": lon0})
    tr.update({"ts_ns": t1, "lat": lat0, "lon": lon0 + dlon_10m})
    tr.update({"ts_ns": t2, "lat": lat0, "lon": lon0 + 2 * dlon_10m})

    est_now = tr.estimate(now_ns=t2)

    mps = est_now["est_vel"]["mps"]
    heading = est_now["est_vel"]["heading_deg"]
    conf = est_now["confidence"]

    # Speed around 10 m/s with ±20% tolerance
    assert 8.0 <= mps <= 12.0
    # Roughly east
    assert heading is not None
    assert 80.0 <= heading <= 100.0
    # Confidence in (0, 1]
    assert 0.0 < conf <= 1.0

    # Confidence should decay after a time gap
    est_later = tr.estimate(now_ns=t2 + int(5e9))
    assert est_later["confidence"] < conf

