# Dev Contract — FieldEvidence Hackathon POC
File: dev/OBJECTIVES.md

## Mission Outcome
Operator loads a synthetic UAV clip with a parasite launch.  
System auto-detects the launch, emits a SALUTE Protobuf microping (≤80 B), queues if offline, and displays a CoT map pin once network resumes.

---

## Non-Negotiable Acceptance Criteria
- ≥3 synthetic clips each yield ≥1 detected launch event.  
- Microping serialized size ≤80 B (avg ≤75 B, max ≤80 B).  
- Store-and-forward:  
  - **Offline** → micropings saved in SQLite immediately.  
  - **Online** → queue flushed ≤2 s after toggle.  
- CoT pin appears in Streamlit map ≤5 s after event detection.  
- End-to-end demo runs on local laptop (Python 3.10+, CPU-only).  

---

## Inputs / Outputs

### Inputs
- **Video**: `.mp4` synthetic fixed-wing + parasite (30 fps, ≤15 s).  
- **Labels (optional)**: `.csv` with columns `[frame, id, class, x, y, w, h]`.  

### Outputs
- **Event log (`.jsonl`)**  
```json
{"uid":"evt_001","ts":1727702945,"lat":52.52,"lon":13.40,"event_code":1}
```

- **SALUTE microping (Protobuf)**  
≤80 B serialized payload using fixed-width fields:  
```proto
syntax = "proto3";
message SalutePing {
  fixed64 uid        = 1;  // 8 B numeric ID
  fixed64 ts_ns      = 2;  // 8 B timestamp (ns)
  sfixed32 lat_q     = 3;  // 4 B lat *1e7
  sfixed32 lon_q     = 4;  // 4 B lon *1e7
  fixed32 geo_conf_m = 5;  // 4 B confidence (m)
  uint32  event_code = 6;  // launch=1, drop=2, decoy=3
  bytes   hash_pref  = 7;  // 4–8 B SHA256 prefix
}
```

- **CoT pin**: XML/JSON marker rendered on Streamlit/Leaflet map.

---

## Demo Script (≈75 s)
1. Operator plays synthetic clip (parasite separates).  
2. Detector overlays BBoxes; divergence triggers launch event.  
3. Event log + SALUTE microping shown (with byte size).  
4. Network OFF → microping queued in SQLite.  
5. Network ON → microping transmitted; queue drains.  
6. CoT pin pops up on map with correct color for `event_code`.  
7. Operator exports SALUTE bundle (JSON + Proto hex preview).  

---
