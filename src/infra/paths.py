from pathlib import Path

# Repo root (two levels above src/infra)
ROOT = Path(__file__).resolve().parents[2]

# Common output locations
OUT = ROOT / "out"
DB_PATH = OUT / "queue.db"
MONITOR = OUT / "monitor.jsonl"
PINGS = OUT / "pings.bin"
COT_DIR = OUT / "cot"
VIZ_DIR = OUT / "viz"
SNIPS_DIR = OUT / "snips"
SENT_BIN = OUT / "sent.bin"

