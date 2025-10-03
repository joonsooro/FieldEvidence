from __future__ import annotations

import json
import time
from pathlib import Path as _Path
from typing import Dict, List, Optional

import folium
import streamlit as st
import xml.etree.ElementTree as ET

from src.infra.net_sim import set_online, is_online
from src.infra import replay_in_app

MONITOR_PATH = _Path("out/monitor.jsonl")
COT_DIR = _Path("out/cot")


try:
    import pandas as _pd  # type: ignore
    _PD_ERR = None
except Exception as e:  # pragma: no cover - optional dependency
    _pd = None
    _PD_ERR = e


def _load_monitor(path: _Path, max_lines: int = 2000):
    snaps = []
    try:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snaps.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        return []
    return snaps[-max_lines:]


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

    # ---- Session state ----
    if "auto_disrupt" not in st.session_state:
        st.session_state.auto_disrupt = True
    if "refresh_s" not in st.session_state:
        st.session_state.refresh_s = 1.0
    if "replay_attempted" not in st.session_state:
        st.session_state.replay_attempted = False

    # ---- Sidebar controls ----
    st.sidebar.markdown("Synthetic only · Rule-based · Demo hash prefix · No real radios")
    st.sidebar.divider()

    c0, c1 = st.sidebar.columns(2)
    if c0.button("Start Replay", type="primary", use_container_width=True):
        try:
            replay_in_app.start_replay(pattern="5:off,5:on", limit=200, budget_ms=50)
            st.session_state.replay_attempted = True
        except Exception as e:
            st.error(f"Failed to start replay: {e}")

    if c1.button("Stop Replay", use_container_width=True):
        try:
            replay_in_app.stop_replay()
        except Exception as e:
            st.error(f"Failed to stop replay: {e}")

    st.session_state.auto_disrupt = st.sidebar.toggle(
        "Comm Disruption", value=st.session_state.auto_disrupt
    )
    # Apply auto disruption toggle live; when OFF, force system ONLINE
    try:
        replay_in_app.set_auto_disruption(st.session_state.auto_disrupt)
        if not st.session_state.auto_disrupt:
            set_online(True)
    except Exception:
        pass

    # Visual status with colored values; labels in red
    online_now = is_online()
    status_color = "green" if online_now else "red"
    replay_running = replay_in_app.is_running()
    replay_color = "green" if replay_running else "red"
    st.sidebar.markdown(
        f"<span style='color:#d22;font-weight:600;'>Status</span>: "
        f"<span style='color:{status_color};font-weight:700;'>{'ONLINE' if online_now else 'OFFLINE'}</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"<span style='color:#d22;font-weight:600;'>Replay</span>: "
        f"<span style='color:{replay_color};font-weight:700;'>{'running' if replay_running else 'stopped'}</span>",
        unsafe_allow_html=True,
    )

    st.session_state.refresh_s = st.sidebar.slider(
        "Refresh every N seconds", min_value=0.5, max_value=10.0, value=float(st.session_state.refresh_s), step=0.5
    )

    # ---- Title ----
    st.title("Field Evidence – Live Demo")

    # ---- Video Panel ----
    st.subheader("Video")
    viz_dir = _Path("out/viz")
    chosen_mp4: Optional[_Path] = None
    try:
        best = viz_dir / "clip_best.mp4"
        if best.exists():
            chosen_mp4 = best
        elif viz_dir.exists():
            vids = sorted(viz_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            chosen_mp4 = vids[0] if vids else None
    except Exception:
        chosen_mp4 = None
    if chosen_mp4 is not None:
        st.video(str(chosen_mp4))
        st.caption("Video with the highest reliability")
    else:
        st.info("No viz found yet — place clip_best.mp4 or any *.mp4 under out/viz/.")

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
    snaps = _load_monitor(MONITOR_PATH)
    if not snaps:
        st.info("No monitor data yet — start replay to generate out/monitor.jsonl.")
    else:
        last = snaps[-1]
        c1, c2, c3, c4 = st.columns(4)
        # Use actual online/offline state
        c1.metric("Online", "True" if online_now else "False")
        c2.metric("Queue depth", int(last.get("depth", 0)))
        c3.metric("sent/s", int(last.get("sent_1s", 0)))
        c4.metric("p50 e2e (ms)", f"{float(last.get('p50_e2e_ms', 0.0)):.0f}")

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

    # ---- Auto refresh ----
    time.sleep(float(st.session_state.refresh_s))
    st.rerun()

if __name__ == "__main__":
    main()
