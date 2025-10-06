from __future__ import annotations

import json
import base64
import logging
import os
import time
from pathlib import Path as _Path
from typing import Dict, List, Optional

import folium
import streamlit as st
import xml.etree.ElementTree as ET

from src.infra import replay_in_app
from src.infra import paths

# Use shared absolute paths so CLI and UI write/read the same files
MONITOR_PATH = paths.MONITOR
SNIPS_DIR = paths.SNIPS_DIR
VIZ_DIR = paths.VIZ_DIR
COT_DIR = paths.COT_DIR


try:
    import pandas as _pd  # type: ignore
    _PD_ERR = None
except Exception as e:  # pragma: no cover - optional dependency
    _pd = None
    _PD_ERR = e


def _load_monitor(path: _Path, max_lines: int = 1000):
    snaps: List[Dict[str, object]] = []
    if not path.exists():
        return snaps
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snaps.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    # keep only last max_lines
    snaps = snaps[-max_lines:]
    # sort by t_ns and de-duplicate by t_ns
    keyed: Dict[int, Dict[str, object]] = {}
    for s in snaps:
        try:
            t = int(s.get("t_ns", 0))
        except Exception:
            continue
        keyed[t] = s
    ordered = [keyed[k] for k in sorted(keyed.keys())]
    return ordered


def _load_monitor_last(path: _Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    try:
        *_, last = path.read_text(encoding="utf-8").splitlines()
        return json.loads(last)
    except Exception:
        return None


def _latest_mp4_under(dir_path: _Path) -> Optional[_Path]:
    try:
        if dir_path.exists():
            vids = sorted(dir_path.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            return vids[0] if vids else None
    except Exception:
        pass
    return None

def _render_video(path: _Path, height: int = 360) -> None:
    """Render an mp4 reliably (base64 data URL fallback for autoplay)."""
    log = logging.getLogger("ui.video")
    try:
        # Temporarily prefer native Streamlit video widget for reliability and smaller payloads.
        # Use env USE_HTML_EMBED=1 to switch back to HTML data-URI embed for autoplay testing.
        if not os.environ.get("USE_HTML_EMBED", "").strip():
            st.video(str(path))
            log.info("Rendered via st.video: %s", path)
            return

        import base64
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        log.info("Rendering via HTML embed: path=%s bytes=%d", path, len(data))
        html = (
            f"<video style='width:100%;height:auto;' autoplay muted playsinline controls>"
            f"<source src='data:video/mp4;base64,{data}' type='video/mp4'>"
            f"</video>"
        )
        st.components.v1.html(html, height=height)
    except Exception as e:
        log.exception("HTML embed failed; falling back to st.video: %s", e)
        st.video(str(path))


def _status_badge(online: bool) -> str:
    color = "#16a34a" if online else "#dc2626"
    text = "Online" if online else "Offline"
    return f"<span style='color:{color};font-weight:700'>{text}</span>"


def _latest_snip() -> Optional[_Path]:
    if not SNIPS_DIR.exists():
        return None
    vids = sorted(SNIPS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return vids[0] if vids else None


def _color_for_event_code(code: int) -> str:
    return {1: "red", 2: "orange", 3: "blue"}.get(int(code), "gray")


def _parse_cot_file(path: _Path) -> Optional[Dict[str, object]]:
    try:
        txt = path.read_text(encoding="utf-8")
        root = ET.fromstring(txt)
        if root.tag != "event":
            return None

        uid = root.attrib.get("uid", "")
        t = root.attrib.get("time", "")
        pt = root.find("point")
        if pt is None:
            return None
        lat = float(pt.attrib.get("lat", "0") or 0.0)
        lon = float(pt.attrib.get("lon", "0") or 0.0)

        # remarks example: "event_code=1 hash_len=6"
        ec = None
        detail = root.find("detail")
        if detail is not None:
            remarks = detail.find("remarks")
            if remarks is not None and remarks.text:
                txt = remarks.text.strip()
                for part in txt.split():
                    if part.startswith("event_code="):
                        try:
                            ec = int(part.split("=", 1)[1])
                        except Exception:
                            pass
                        break
        event_code = int(ec) if ec is not None else 0

        # Derive uid_hex if uid like "uid_<int>"
        uid_hex = None
        if uid.startswith("uid_"):
            try:
                uid_int = int(uid.split("_", 1)[1])
                uid_hex = hex(uid_int)
            except Exception:
                uid_hex = None

        return {
            "uid": uid,
            "uid_hex": uid_hex,
            "time": t,
            "lat": lat,
            "lon": lon,
            "event_code": event_code,
            "path": str(path),
        }
    except Exception:
        return None


def _load_cot_events(dir_path: _Path) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    if not dir_path.exists():
        return events
    for p in sorted(dir_path.glob("*.xml")):
        ev = _parse_cot_file(p)
        if ev is not None:
            events.append(ev)
    return events


def _tail_jsonl(path: _Path, n: int = 100) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        out: List[Dict[str, object]] = []
        for line in lines[-int(n):]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _render_map(events: List[Dict[str, object]], selected_codes: List[int]) -> str:
    pts = [(float(e["lat"]), float(e["lon"])) for e in events if (not selected_codes or int(e.get("event_code", 0)) in selected_codes)]
    if pts:
        clat = sum(p[0] for p in pts) / len(pts)
        clon = sum(p[1] for p in pts) / len(pts)
    else:
        clat, clon = 0.0, 0.0

    # Zoom closer when we have events (~12); fallback to 6
    zoom_start = 12 if pts else 6
    m = folium.Map(location=[clat, clon], zoom_start=zoom_start)

    for e in events:
        ec = int(e.get("event_code", 0))
        if selected_codes and ec not in selected_codes:
            continue
        lat = float(e["lat"])
        lon = float(e["lon"])
        uid = str(e.get("uid", ""))
        t = str(e.get("time", ""))
        color = _color_for_event_code(ec)
        popup = folium.Popup(html=f"<b>{uid}</b><br/>time={t}<br/>code={ec}", max_width=250)
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(m)

    # Return HTML string for embedding
    return m.get_root().render()


def main() -> None:
    st.set_page_config(page_title="Field Evidence Demo", layout="wide")

    # ---- Session state & background threads ----
    if "auto_disrupt" not in st.session_state:
        # default auto disruption ON at first render
        st.session_state.auto_disrupt = True
    # Ensure background scheduler/toggler are running per requested state
    try:
        replay_in_app.ensure_toggler_running(st.session_state.auto_disrupt)
        replay_in_app.ensure_replay_running(limit=200, budget_ms=50)
    except Exception:
        pass

    # ---- Sidebar controls ----
    st.sidebar.markdown("Synthetic only · Rule-based · Demo hash prefix · No real radios")
    st.sidebar.divider()

    # Comm disruption toggle (auto OFF/ON every 5s)
    st.sidebar.subheader("Comm disruption")
    auto = st.sidebar.toggle("Comm Disruption", value=st.session_state.auto_disrupt)
    if auto != st.session_state.auto_disrupt:
        st.session_state.auto_disrupt = auto
        try:
            replay_in_app.ensure_toggler_running(bool(auto))
        except Exception:
            pass

    # Optional: reset monitor for a clean session
    if st.sidebar.button("Reset monitor"):
        try:
            if MONITOR_PATH.exists():
                MONITOR_PATH.write_text("", encoding="utf-8")
            # Force viz fallback until a new snip appears
            st.session_state["_force_viz"] = True
        except Exception:
            pass

    # Visual status based on latest snapshot
    last = _load_monitor(MONITOR_PATH, max_lines=1)
    last = last[-1] if last else None
    online_flag = bool(last.get("online")) if last else False
    st.sidebar.markdown(
        f"Status: {_status_badge(online_flag)}",
        unsafe_allow_html=True,
    )

    # Fixed 1s refresh cadence to match 1 Hz monitor

    # ---- Title ----
    st.title("Field Evidence – Live Demo")

    

    # ---- Events Panel ----
    st.subheader("Events")
    events = _load_cot_events(COT_DIR)
    if not events:
        st.info("No CoT events yet — start replay to populate out/cot/.")
    else:
        # Show latest ~200
        ev_show = events[-200:]
        rows = [
            {
                "time": e.get("time"),
                "uid(hex)": e.get("uid_hex") or e.get("uid"),
                "lat": e.get("lat"),
                "lon": e.get("lon"),
                "event_code": e.get("event_code"),
            }
            for e in ev_show
        ]
        if _pd is None:
            st.warning(f"Pandas unavailable ({_PD_ERR}). Showing raw entries.")
            st.write(rows)
        else:
            df = _pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- Map Panel ----
    st.subheader("Map")
    # Build simple palette by event_code
    # Allow filtering via a small multiselect when events exist
    codes_present = sorted({int(e.get("event_code", 0)) for e in events})
    selected_codes: List[int] = []
    if codes_present:
        selected_codes = st.multiselect("Filter event_code", options=codes_present, default=codes_present)
    map_html = _render_map(events, selected_codes)
    st.components.v1.html(map_html, height=520, scrolling=False)

    # ---- Queue Monitor & Stats ----
    st.subheader("Queue Monitor & Stats")
    snaps = _load_monitor(MONITOR_PATH, max_lines=1000)
    if not snaps:
        st.info("No monitor data yet — start replay to generate out/monitor.jsonl.")
    else:
        last_row = snaps[-1]
        st.markdown(
            f"Online: {_status_badge(bool(last_row.get('online', False)))}",
            unsafe_allow_html=True,
        )
        c2, c3, c4 = st.columns(3)
        c2.metric("Queue depth", int(last_row.get("depth", 0)))
        c3.metric("sent/s", int(last_row.get("sent_1s", 0)))
        c4.metric("p50 e2e (ms)", f"{float(last_row.get('p50_e2e_ms', 0.0)):.0f}")

        cols = [
            "t_ns",
            "depth",
            "sent_tot",
            "sent_1s",
            "p50_e2e_ms",
            "p95_e2e_ms",
            "p50_drain_ms",
            "p95_drain_ms",
            "online",
        ]
        table_rows = [{k: s.get(k) for k in cols} for s in snaps[-200:]]
        if _pd is None:
            st.write(table_rows)
        else:
            df = _pd.DataFrame(table_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- Bottom Video Panel ----
    st.subheader("Video")

    # Choose what to show:
    # 1) If there is any snip, show the newest snip.
    # 2) Else if _force_viz is set OR no snips exist, show newest viz.
    # 3) Else show an info message.

    snip = _latest_mp4_under(SNIPS_DIR)
    viz  = _latest_mp4_under(VIZ_DIR)
    log = logging.getLogger("ui.video")

    use_viz = bool(st.session_state.get("_force_viz")) and viz is not None
    # Debug visibility in UI and terminal
    st.caption(f"DEBUG: snip={snip} viz={viz} use_viz={use_viz}")
    try:
        log.info("Bottom Video Panel select snip=%s viz=%s _force_viz=%s use_viz=%s", snip, viz, st.session_state.get("_force_viz"), use_viz)
    except Exception:
        pass
    if snip is not None and not use_viz:
        # Show the most recent snip (created on event)
        _render_video(snip, height=360)
        st.caption("Video generated at incident time (±2 s)")
    elif viz is not None:
        _render_video(viz, height=360)
        st.caption("Fallback: latest viz")
    else:
        st.info("No video available yet. When an incident occurs, a snip will be generated here.")

    # Once we have shown a snip, clear the viz-forced flag
    if snip is not None and st.session_state.get("_force_viz"):
        st.session_state["_force_viz"] = False
        try:
            log.info("Cleared _force_viz after rendering snip %s", snip)
        except Exception:
            pass

    # ---- C2 Panel ----
    # Appended section: optional C2 pipeline UI. Degrades gracefully.
    st.subheader("C2 Panel")
    st.caption("SIMULATION ONLY / Human-in-loop / Synthetic")

    # ---- Label Mode Fallback Badge (optional) ----
    try:
        from pathlib import Path as _P
        _label_flag = _P("out/label_mode.on")
        if _label_flag.exists():
            st.warning("🟡 Label Mode Active — using fallback events", icon="⚠️")
    except Exception:
        pass

    # Optional metrics import guard; degrades to no-op if unavailable
    try:
        from src.c2.metrics import Metrics, measure_decision  # type: ignore
    except Exception:
        Metrics = None  # type: ignore
        def measure_decision(fn, *a, **k):  # type: ignore
            return fn(*a, **k), 0.0

    # Safe, optional imports inside the panel to avoid impacting app load.
    _C2_OK = True
    try:
        from pathlib import Path as _P
        import yaml as _yaml  # type: ignore
        from src.c2.estimator import Tracker as _Tracker  # type: ignore
        from src.c2.prioritizer import compute_risk as _compute_risk  # type: ignore
        from src.c2.recommendation import make_recommendation as _make_reco  # type: ignore
        from src.c2.scheduler import approve_and_dispatch as _approve  # type: ignore
    except Exception:
        _C2_OK = False

    if not _C2_OK:
        st.info("C2 modules not available. Panel is disabled.")
    else:
        # Initialize panel-local session state
        if "c2_log" not in st.session_state:
            st.session_state.c2_log = []  # type: ignore[attr-defined]
        if "c2_last_dispatch" not in st.session_state:
            st.session_state.c2_last_dispatch = None  # type: ignore[attr-defined]

        # Load configuration (optional); fallback to sensible defaults
        _cfg_path = _P("config.yaml")
        _cfg_default = {
            "asset_speed_mps": 25.0,
            "weights": {"intent": 0.5, "proximity": 0.3, "recent": 0.2},
            "thresholds": {"IMMEDIATE": 0.8, "SUSPECT": 0.5},
            "target": {"lat": 52.5200, "lon": 13.4050},
        }
        _cfg = dict(_cfg_default)
        try:
            if _cfg_path.exists():
                _loaded = _yaml.safe_load(_cfg_path.read_text(encoding="utf-8")) or {}
                if isinstance(_loaded, dict):
                    # Shallow-merge
                    _cfg.update(_loaded)
        except Exception:
            pass

        _target = dict(_cfg.get("target", {}))
        _t_lat = float(_target.get("lat", 52.5200))
        _t_lon = float(_target.get("lon", 13.4050))

        # Load recent normalized events for the left column
        _feed_path = _P("out/c2_feed.jsonl")
        _events: list[dict] = []
        if _feed_path.exists():
            try:
                from collections import deque as _deque  # local, cheap
                _dq = _deque(maxlen=50)
                with _feed_path.open("r", encoding="utf-8") as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _dq.append(json.loads(_line))
                        except Exception:
                            continue
                _events = list(_dq)
            except Exception:
                _events = []

        # 3 columns: feed, estimation, risk/recommendation
        _left, _mid, _right = st.columns(3)

        # Left: c2 feed table (or info)
        with _left:
            if not _events:
                st.info("C2 feed not available yet.")
            else:
                _cols = [
                    "event_id",
                    "uid",
                    "ts_ns",
                    "lat",
                    "lon",
                    "event_code",
                    "conf",
                    "intent_score",
                ]
                _rows = [{k: e.get(k) for k in _cols} for e in _events]
                try:
                    if _pd is None:
                        st.write(_rows)
                    else:
                        _df = _pd.DataFrame(_rows)
                        st.dataframe(_df, use_container_width=True, hide_index=True)
                except Exception:
                    st.write(_rows)

        # Middle: latest estimation via Tracker on last K events
        _estimation: dict | None = None
        _est_ms: float | None = None
        _last_event: dict | None = _events[-1] if _events else None
        with _mid:
            if not _events:
                st.info("Insufficient events for estimation.")
            else:
                try:
                    def _do_est():
                        _tracker = _Tracker(window_sec=30.0, decay_tau_sec=20.0)
                        # Order last K by ts_ns to ensure monotonic updates
                        _K = 10
                        _by_ts = sorted(_events[-_K:], key=lambda e: int(e.get("ts_ns", 0)))
                        for _e in _by_ts:
                            _tracker.update({
                                "ts_ns": int(_e.get("ts_ns", 0)),
                                "lat": float(_e.get("lat", 0.0)),
                                "lon": float(_e.get("lon", 0.0)),
                                "conf": float(_e.get("conf", 1.0)) if _e.get("conf") is not None else 1.0,
                            })
                        _now_ns = int(_by_ts[-1].get("ts_ns", 0)) if _by_ts else 0
                        return _tracker.estimate(now_ns=_now_ns)

                    (_estimation, _est_ms) = measure_decision(_do_est)
                    try:
                        if Metrics is not None and _est_ms is not None:
                            Metrics().record_estimate_ms(float(_est_ms))
                    except Exception:
                        pass

                    _pos = _estimation.get("est_pos", {})
                    _vel = _estimation.get("est_vel", {})
                    _conf = float(_estimation.get("confidence", 0.0))

                    _m1, _m2 = st.columns(2)
                    with _m1:
                        st.metric("Est Lat", f"{float(_pos.get('lat', 0.0)):.5f}")
                        st.metric("Velocity (m/s)", f"{float(_vel.get('mps', 0.0)):.1f}")
                    with _m2:
                        st.metric("Est Lon", f"{float(_pos.get('lon', 0.0)):.5f}")
                        _hdg = _vel.get("heading_deg", None)
                        st.metric("Heading (deg)", "—" if _hdg is None else f"{float(_hdg):.0f}")
                    st.caption(f"Confidence: {_conf:.2f}")
                except Exception:
                    _estimation = None
                    st.info("Insufficient events for estimation.")

        # Right: risk and recommendation with actions
        _risk: dict | None = None
        _reco: dict | None = None
        with _right:
            if _estimation is not None and _last_event is not None:
                # Measure per-stage and total latencies
                _metrics = Metrics() if Metrics is not None else None

                # risk
                try:
                    def _do_risk():
                        return _compute_risk(_estimation, _t_lat, _t_lon, _cfg)
                    (_risk, _risk_ms) = measure_decision(_do_risk)
                    try:
                        if _metrics is not None:
                            _metrics.record_risk_ms(float(_risk_ms))
                    except Exception:
                        pass
                except Exception:
                    try:
                        _risk = _compute_risk(_estimation, _t_lat, _t_lon, _cfg)
                    except Exception:
                        _risk = None
                    _risk_ms = 0.0

                # reco
                try:
                    def _do_reco():
                        return _make_reco(_estimation, _last_event, _t_lat, _t_lon, _cfg)
                    (_reco, _reco_ms) = measure_decision(_do_reco)
                    try:
                        if _metrics is not None:
                            _metrics.record_reco_ms(float(_reco_ms))
                    except Exception:
                        pass
                except Exception:
                    try:
                        _reco = _make_reco(_estimation, _last_event, _t_lat, _t_lon, _cfg)
                    except Exception:
                        _reco = None
                    _reco_ms = 0.0

                # total envelope (approximate as sum of stages if available)
                try:
                    _total_ms = float((_est_ms or 0.0) + float(_risk_ms or 0.0) + float(_reco_ms or 0.0))
                    if _metrics is not None:
                        _metrics.record_total_ms(_total_ms)
                except Exception:
                    pass

                # Show summary card
                if _risk is not None:
                    _status = str(_risk.get("status", "")).upper()
                    _score = float(_risk.get("score", 0.0))
                    st.markdown(f"**Status:** {_status}  ·  **Score:** {_score:.2f}")
                    _f = dict(_risk.get("factors", {}))
                    _dist = float(_f.get("distance_m", 0.0))
                    _tti = float(_f.get("tti_sec", 0.0))
                    _cs = float(_f.get("closing_speed_mps", 0.0))
                    st.caption(
                        f"distance={_dist:.0f} m · tti={_tti:.1f} s · closing={_cs:.1f} m/s"
                    )

                if _reco is None:
                    st.info("No immediate recommendation; SUSPECT may produce OBSERVE recommendation")
                # Action buttons
                _cA, _cB = st.columns(2)
                with _cA:
                    _approve_clicked = st.button("Approve", use_container_width=True)
                with _cB:
                    _ignore_clicked = st.button("Ignore", use_container_width=True)

                if _approve_clicked:
                    try:
                        if _reco is None or str(_reco.get("action", {}).get("type", "")).upper() != "INTERCEPT":
                            st.warning("No actionable recommendation (not IMMEDIATE).")
                        else:
                            _out = _approve(
                                _reco,
                                _estimation,  # type: ignore[arg-type]
                                _cfg,
                                audit_path=_P("out/c2_audit.jsonl"),
                            )
                            st.session_state.c2_last_dispatch = _out  # type: ignore[attr-defined]
                            try:
                                _line = f"APPROVED {_reco.get('event_id')} → {_out.get('asset_id')} ETA={float(_out.get('eta_sec', 0.0)):.1f}s"
                                st.session_state.c2_log.append(_line)  # type: ignore[attr-defined]
                                st.session_state.c2_log = st.session_state.c2_log[-20:]  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            st.toast("SIMULATION ONLY: dispatch scheduled")
                    except Exception as _e:
                        st.warning(f"Dispatch failed: {_e}")

                if _ignore_clicked:
                    try:
                        _ev_id = (_last_event or {}).get("event_id")
                        st.session_state.c2_log.append(f"IGNORED {_ev_id}")  # type: ignore[attr-defined]
                        st.session_state.c2_log = st.session_state.c2_log[-20:]  # type: ignore[attr-defined]
                    except Exception:
                        pass
            else:
                st.info("Waiting for events to compute risk.")

        # Map: always render
        try:
            _center_lat, _center_lon = _t_lat, _t_lon
            _zoom = 6
            if _estimation is not None:
                _pos = _estimation.get("est_pos", {})
                _center_lat = float(_pos.get("lat", _t_lat))
                _center_lon = float(_pos.get("lon", _t_lon))
                _zoom = 12

            _m = folium.Map(location=[_center_lat, _center_lon], zoom_start=_zoom)
            # Target asset marker
            try:
                folium.Marker(
                    location=(_t_lat, _t_lon),
                    tooltip="Asset",
                    icon=folium.Icon(color="green", icon="flag"),
                ).add_to(_m)
            except Exception:
                pass
            # Estimated position marker
            if _estimation is not None:
                try:
                    _pos = _estimation.get("est_pos", {})
                    folium.Marker(
                        location=(float(_pos.get("lat", _t_lat)), float(_pos.get("lon", _t_lon))),
                        tooltip="Est Pos",
                        icon=folium.Icon(color="red"),
                    ).add_to(_m)
                except Exception:
                    pass
            # Intercept marker + line if available
            try:
                _ld = st.session_state.c2_last_dispatch  # type: ignore[attr-defined]
                if _ld and isinstance(_ld, dict):
                    _intc = _ld.get("intercept", {})
                    _ilat = float(_intc.get("lat"))
                    _ilon = float(_intc.get("lon"))
                    _eta = float(_ld.get("eta_sec", 0.0))
                    folium.Marker(
                        location=(_ilat, _ilon),
                        tooltip=f"Intercept (ETA ~{_eta:.0f}s)",
                        icon=folium.Icon(color="blue", icon="target"),
                    ).add_to(_m)
                    if _estimation is not None:
                        _pos = _estimation.get("est_pos", {})
                        folium.PolyLine([
                            (float(_pos.get("lat", _t_lat)), float(_pos.get("lon", _t_lon))),
                            (_ilat, _ilon),
                        ], color="blue", weight=2, opacity=0.8).add_to(_m)
            except Exception:
                pass

            _map_html = _m.get_root().render()
            st.components.v1.html(_map_html, height=420, scrolling=False)
        except Exception:
            st.info("Map unavailable.")

        # Rolling mini log
        try:
            _log_lines = st.session_state.c2_log  # type: ignore[attr-defined]
            if _log_lines:
                st.caption("C2 Log (last 20)")
                st.code("\n".join(_log_lines))
        except Exception:
            pass

        # Tiny metrics row under the C2 Panel (additive UI)
        try:
            import json as _json
            from pathlib import Path as _PathLocal

            _colA, _colB, _colC = st.columns(3)
            _metrics_path = _PathLocal("out/metrics.json")
            _ping_sizes_path = _PathLocal("out/ping_sizes.json")
            _monitor_path = _PathLocal("out/monitor.jsonl")

            # Decision/Total latency p95
            if _metrics_path.exists():
                try:
                    _m = _json.loads(_metrics_path.read_text(encoding="utf-8"))
                    _p95 = (_m.get("decision_latency") or {}).get("p95_ms")
                except Exception:
                    _p95 = None
                _colA.metric("Total p95 (ms)", f"{float(_p95):.0f}" if _p95 else "n/a")
            else:
                _colA.caption("Total p95: n/a")

            # Payload size p95/max
            if _ping_sizes_path.exists():
                try:
                    _ps = _json.loads(_ping_sizes_path.read_text(encoding="utf-8"))
                    _p95_sz, _max_sz = _ps.get("p95"), _ps.get("max")
                    _txt = f"{int(_p95_sz)} / {int(_max_sz)}" if (_p95_sz and _max_sz) else "n/a"
                except Exception:
                    _txt = "n/a"
                _colB.metric("Payload p95/max (B)", _txt)
            else:
                _colB.caption("Payload: n/a")

            # Recovery p95
            _recov = None
            if _monitor_path.exists():
                try:
                    _last = _json.loads(_monitor_path.read_text().splitlines()[-1])
                    _recov = _last.get("p95_drain_ms")
                except Exception:
                    _recov = None
            _colC.metric("Recovery p95 (ms)", f"{float(_recov):.0f}" if _recov else "n/a")

        except Exception:
            st.caption("Metrics: n/a")

    # Keep steady 1 Hz refresh without extra UI controls
    time.sleep(1.0)
    st.rerun()

if __name__ == "__main__":
    main()
