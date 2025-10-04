# Mock C2 System — 24-Hour Development & Integration Plan
*(Field-ready MVP for EDTH Hackathon, powered by Codex)*

---

## 🎯 Final Acceptance Criteria

- **90-second demo loop:**  
  Event → SALUTE (≤80 B) → (offline 5 s) → recovery ≤ 2 s → C2 recommendation → human approval → simulated dispatch (ETA + map intercept)
- **Metrics displayed:** payload p95/max, recovery p95, decision latency < 1 s
- **Badges:** `SIMULATION ONLY / Human-in-loop / Synthetic`
- **Fallback:** if vision fails → labels mode with visible UI badge

---

## 🕒 Schedule Overview (6 Blocks × 4 hours = 24 h)

| Block | Module | Main Deliverables | Owner |
|-------|---------|------------------|--------|
| 1 | Receiver & Event Feed | `src/c2/mock_c2.py` | BE |
| 2 | Estimator (Linear Predictor) | `src/c2/estimator.py` | BE |
| 3 | Risk & Recommendation | `src/c2/prioritizer.py`, `src/c2/recommendation.py` | BE |
| 4 | Scheduler & Audit | `src/c2/scheduler.py`, `out/c2_audit.jsonl` | BE |
| 5 | Streamlit UI Panel | `src/ui/app.py` (C2 panel) | FE |
| 6 | Stability & Rehearsal | metrics + `run_demo.sh` | PM / ALL |

---

## ✅ Core Dependencies