# src/c2/state.py
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from .estimator import Tracker
from .prioritizer import compute_risk
from .recommendation import make_recommendation

@dataclass
class TargetState:
    uid: int
    tracker: Tracker = field(default_factory=Tracker)
    last_risk: Optional[dict] = None
    last_reco: Optional[dict] = None
    re_eval_at_ns: Optional[int] = None
    history: List[dict] = field(default_factory=list)  # keep small timeline

class C2State:
    def __init__(self, promote_after_sec: float = 10.0, require_new_event: bool = True):
        self.targets: Dict[int, TargetState] = {}
        self.promote_after_ns = int(promote_after_sec * 1e9)
        self.require_new_event = require_new_event

    def ingest_event(self, ev: dict, cfg: dict, target_lat: float, target_lon: float):
        uid = int(ev["uid"])
        ts_ns = int(ev["ts_ns"])
        st = self.targets.setdefault(uid, TargetState(uid))
        # Update tracker
        st.tracker.update({"ts_ns": ts_ns, "lat": ev["lat"], "lon": ev["lon"]})

        # Compute risk/recommendation from latest estimate
        est = st.tracker.estimate(ts_ns)
        risk = compute_risk(est, target_lat, target_lon, cfg)
        # Produce recommendations for SUSPECT/IMMEDIATE (OBSERVE/INTERCEPT)
        from .recommendation import make_recommendation
        reco = make_recommendation(est, ev, target_lat, target_lon, cfg)

        st.last_risk, st.last_reco = risk, reco
        st.history.append({"ts_ns": ts_ns, "risk": risk, "reco": reco})

        # If SUSPECT, set re-evaluation timer
        if risk["status"] == "SUSPECT" and st.re_eval_at_ns is None:
            st.re_eval_at_ns = ts_ns + self.promote_after_ns

    def tick(self, now_ns: int, cfg: dict, target_lat: float, target_lon: float):
        # Re-evaluate SUSPECT targets whose timer expired
        for st in self.targets.values():
            if st.re_eval_at_ns and now_ns >= st.re_eval_at_ns:
                # Option: skip if no new events arrived
                if self.require_new_event:
                    # Check if history has ts_ns after the re-eval window
                    newest_ts = max(h["ts_ns"] for h in st.history) if st.history else 0
                    if newest_ts < st.re_eval_at_ns:
                        continue  # No new events → defer re-evaluation

                # Re-evaluate at the most recent observation time
                ref_ts = max(h["ts_ns"] for h in st.history) if st.history else now_ns
                est = st.tracker.estimate(ref_ts)
                # If confidence has decayed by re-eval time, reflect that as-is
                dummy_event = {"event_id": f"reval#{now_ns}", "uid": st.uid, "ts_ns": ref_ts}
                risk = compute_risk(est, target_lat, target_lon, cfg)
                reco = make_recommendation(est, dummy_event, target_lat, target_lon, cfg)
                st.last_risk, st.last_reco = risk, reco
                st.history.append({"ts_ns": now_ns, "risk": risk, "reco": reco})
                # Clear timer if IMMEDIATE
                if risk["status"] == "IMMEDIATE":
                    st.re_eval_at_ns = None
