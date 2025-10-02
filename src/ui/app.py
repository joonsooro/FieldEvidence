from __future__ import annotations

import _json
import time
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

import folium
import streamlit as st
import xml.etree.ElementTree as ET

from src.infra.net_sim import set_online, is_online

MONITOR_PATH = _Path("out/monitor.jsonl")
COT_DIR = _Path("out/cot")


try:
    import pandas as _pd
    _PD_ERR = None
except Exception as e:
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
                    snaps.append(_json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        return []
    return snaps[-max_lines:]


def _color_for_event_code(code: int) -> str:
    return {1: "red", 2: "orange", 3: "blue"}.get(int(code), "gray")


def _parse_cot_file(path: Path) -> Optional[Dict[str, object]]:
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

        return {
            "uid": uid,
            "time": t,
            "lat": lat,
            "lon": lon,
            "event_code": event_code,
            "path": str(path),
        }
    except Exception:
        return None


def _load_cot_events(dir_path: Path) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    if not dir_path.exists():
        return events
    for p in sorted(dir_path.glob("*.xml")):
        ev = _parse_cot_file(p)
        if ev is not None:
            events.append(ev)
    return events


def _tail_jsonl(path: Path, n: int = 100) -> List[Dict[str, object]]:
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

    m = folium.Map(location=[clat, clon], zoom_start=6)

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
    st.set_page_config(page_title="CoT Map + Queue Monitor", layout="wide")

    # Sidebar: network controls
    st.sidebar.header("Network")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Online"):
        set_online(True)
    if c2.button("Offline"):
        set_online(False)
    st.sidebar.write(f"Status: {'ONLINE' if is_online() else 'offline'}")

    refresh_sec = st.sidebar.slider("Refresh interval (s)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)

    # Load CoT events and unique event codes
    events = _load_cot_events(COT_DIR)
    codes_present = sorted({int(e.get("event_code", 0)) for e in events})
    selected_codes = st.sidebar.multiselect("Filter: event_code", options=codes_present, default=codes_present)

    st.title("TAK CoT Map (Demo)")

    # Map view
    map_html = _render_map(events, selected_codes)
    st.subheader("Map")
    st.components.v1.html(map_html, height=520, scrolling=False)

    st.subheader("Queue Monitor")
    # ---- Queue Monitor (safe, works even without pandas) ----
    snaps = _load_monitor(MONITOR_PATH)
    if not snaps:
        st.info("No monitor data yet — run the replay to generate `out/monitor.jsonl`.")
    else:
        last = snaps[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Online", "True" if last.get("online") else "False")
        c2.metric("Queue depth", int(last.get("depth", 0)))
        c3.metric("sent/s", int(last.get("sent_1s", 0)))
        c4.metric("p50 e2e (ms)", f'{float(last.get("p50_e2e_ms", 0.0)):.0f}')

        proj = [{
            "t_ns": s.get("t_ns"),
            "depth": s.get("depth"),
            "sent_tot": s.get("sent_tot"),
            "sent_1s": s.get("sent_1s"),
            "p50_e2e_ms": s.get("p50_e2e_ms"),
            "p95_e2e_ms": s.get("p95_e2e_ms"),
            "online": s.get("online"),
        } for s in snaps][-200:]

        if _pd is None:
            st.warning(f"Pandas unavailable ({_PD_ERR}). Showing raw records.")
            st.write(proj)
        else:
            df = _pd.DataFrame(proj)
            st.dataframe(df, use_container_width=True, hide_index=True)

        
    rows = _tail_jsonl(MONITOR_PATH, n=100)
    if rows:
        # Display selected columns in a compact table
        cols = ["t_ns", "depth", "sent_tot", "sent_1s", "p50_e2e_ms", "p50_drain_ms", "online"]
        # Gracefully project available keys
        proj = [{k: r.get(k) for k in cols} for r in rows]
        st.dataframe(proj, use_container_width=True, hide_index=True)
    else:
        st.write("No monitor data yet. Run replay to populate out/monitor.jsonl.")

    # Stats
    st.subheader("Stats")
    depth_now = rows[-1].get("depth") if rows else None
    sent_tot = rows[-1].get("sent_tot") if rows else None
    last_event_time = events[-1].get("time") if events else None
    st.write(
        f"Queue depth now: {depth_now if depth_now is not None else '-'}  |  "
        f"Sent total: {sent_tot if sent_tot is not None else '-'}  |  "
        f"Pings in out/cot/: {len(events)}  |  "
        f"Last event: {last_event_time if last_event_time else '-'}"
    )

    # Auto-refresh
    time.sleep(max(0.1, float(refresh_sec)))
    st.experimental_rerun()


if __name__ == "__main__":
    main()

