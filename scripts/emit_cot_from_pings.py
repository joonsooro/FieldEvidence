from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infra.cot_encode import microping_to_cot_xml, write_cot_file
from src.wire import salute_pb2
from src.wire.codec import read_ld_stream


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert SALUTE micropings to minimal CoT XML files")
    p.add_argument("--pings", required=True, help="Varint-framed SALUTE binary stream (out/pings.bin)")
    p.add_argument("--out_dir", required=True, help="Output directory for CoT XMLs (e.g., out/cot)")
    p.add_argument("--max", type=int, default=100, help="Max pings to convert (0=all)")
    p.add_argument("--callsign", default="UAV-1", help="Contact callsign for CoT detail")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bin_path = Path(args.pings)
    out_dir = Path(args.out_dir)

    if not bin_path.exists():
        print(f"[ERR] pings file not found: {bin_path}")
        return 2

    written: List[Path] = []
    try:
        with bin_path.open("rb") as fp:
            for i, blob in enumerate(read_ld_stream(fp)):
                if args.max and i >= int(args.max):
                    break
                msg = salute_pb2.SalutePing()
                msg.ParseFromString(blob)
                xml = microping_to_cot_xml(msg, callsign=str(args.callsign))
                p = write_cot_file(xml, out_dir, uid=msg.uid, ts_ns=msg.ts_ns)
                written.append(p)
    except Exception as e:
        print(f"[ERR] failed to convert pings: {e}")
        return 3

    print(f"[OK] wrote {len(written)} CoT file(s) to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

