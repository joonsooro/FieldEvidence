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
        # 추정기 업데이트
        st.tracker.update({"ts_ns": ts_ns, "lat": ev["lat"], "lon": ev["lon"]})

        # 최신 추정으로 리스크/권고 산출
        est = st.tracker.estimate(ts_ns)
        risk = compute_risk(est, target_lat, target_lon, cfg)
        # 권고는 SUSPECT/IMMEDIATE 모두 생성(OBSERVE/INTERCEPT)
        from .recommendation import make_recommendation
        reco = make_recommendation(est, ev, target_lat, target_lon, cfg)

        st.last_risk, st.last_reco = risk, reco
        st.history.append({"ts_ns": ts_ns, "risk": risk, "reco": reco})

        # SUSPECT면 재평가 타이머 세팅
        if risk["status"] == "SUSPECT" and st.re_eval_at_ns is None:
            st.re_eval_at_ns = ts_ns + self.promote_after_ns

    def tick(self, now_ns: int, cfg: dict, target_lat: float, target_lon: float):
        # 타이머 만료된 SUSPECT 대상 재평가
        for st in self.targets.values():
            if st.re_eval_at_ns and now_ns >= st.re_eval_at_ns:
                # 옵션: 새 이벤트가 없었다면 패스
                if self.require_new_event:
                    # last history의 ts_ns가 re_eval window 이후로 들어왔는지 확인
                    newest_ts = max(h["ts_ns"] for h in st.history) if st.history else 0
                    if newest_ts < st.re_eval_at_ns:
                        continue  # 새 이벤트 없음 → 재평가 보류

                # 가장 최근 관측 시각으로 재평가
                ref_ts = max(h["ts_ns"] for h in st.history) if st.history else now_ns
                est = st.tracker.estimate(ref_ts)
                # re-eval 시 confidence가 낮아졌다면 그대로 반영됨
                dummy_event = {"event_id": f"reval#{now_ns}", "uid": st.uid, "ts_ns": ref_ts}
                risk = compute_risk(est, target_lat, target_lon, cfg)
                reco = make_recommendation(est, dummy_event, target_lat, target_lon, cfg)
                st.last_risk, st.last_reco = risk, reco
                st.history.append({"ts_ns": now_ns, "risk": risk, "reco": reco})
                # IMMEDIATE이면 타이머 제거
                if risk["status"] == "IMMEDIATE":
                    st.re_eval_at_ns = None