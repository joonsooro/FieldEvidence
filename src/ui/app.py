from __future__ import annotations

import json
import base64
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

    # ---- Video Panel (half width) ----
    st.subheader("Video")
    left, right = st.columns([1, 1])
    with left:
        snip = _latest_snip()
        if snip:
            # Use HTML5 video tag with data URL to ensure autoplay
            try:
                data = base64.b64encode(snip.read_bytes()).decode("ascii")
                html = (
                    f"<video style='width:100%;height:auto;' autoplay muted playsinline controls>"
                    f"<source src='data:video/mp4;base64,{data}' type='video/mp4'>"
                    f"</video>"
                )
                st.components.v1.html(html, height=360)
            except Exception:
                st.video(str(snip))
            st.caption("video with the highest reliability")
        else:
            vids = (
                sorted(VIZ_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
                if VIZ_DIR.exists() else []
            )
            if vids:
                st.video(str(vids[0]))
                st.caption("latest viz (fallback)")
            else:
                st.info("No snips/viz found yet.")
    with right:
        if last:
            d = int(last.get("depth", 0))
            sps = int(last.get("sent_1s", 0))
            p50 = float(last.get("p50_e2e_ms", 0.0))
            st.markdown(f"Online: {_status_badge(online_flag)}", unsafe_allow_html=True)
            st.metric("Queue depth", d)
            st.metric("sent/s", sps)
            st.metric("p50 e2e (ms)", f"{p50:.0f}")
        else:
            st.info("Waiting for monitor…")

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

    # Keep steady 1 Hz refresh without extra UI controls
    time.sleep(1.0)
    st.rerun()

if __name__ == "__main__":
    main()
