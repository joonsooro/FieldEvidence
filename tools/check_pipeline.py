#!/usr/bin/env python
import sys, json
from pathlib import Path

def fail(msg, code=1):
    print(f"❌ {msg}")
    sys.exit(code)

def require_file(p: Path, msg: str):
    if not p.exists():
        fail(msg)

def read_varint(f):
    shift = 0
    out = 0
    while True:
        b = f.read(1)
        if not b:
            return None
        b = b[0]
        out |= (b & 0x7F) << shift
        if not (b & 0x80):
            return out
        shift += 7

def main():
    # 0) Ensure generated protobuf stubs exist (prevents “forgot make proto”)
    for f in ["src/wire/salute_pb2.py", "src/wire/salute_pb2_grpc.py"]:
        require_file(Path(f), f"missing generated file: {f} (run: make proto)")

    # 1) Size report must exist
    sr = Path("out/ping_sizes.json")
    require_file(sr, "size_report missing: out/ping_sizes.json (run: make emit)")
    report = json.loads(sr.read_text())
    print("✅ size_report loaded:", report)

    # 2) Enforce size constraint
    mx = int(report.get("max", 9999))
    if mx > 80:
        fail(f"max size {mx} > 80 bytes", code=2)
    print(f"✅ Microping size constraint OK (max={mx}B)")

    # 3) Decode first two messages to validate framing/fields
    try:
        import src.wire.salute_pb2 as salute_pb2
    except Exception as e:
        fail(f"cannot import salute_pb2: {e}")

    dec = 0
    with open("out/pings.bin", "rb") as f:
        for i in range(2):
            n = read_varint(f)
            if n is None:
                break
            msg = f.read(n)
            ping = salute_pb2.SalutePing()
            ping.ParseFromString(msg)
            print(f"🔎 Ping {i+1}:",
                  dict(uid=ping.uid,
                       ts_ns=ping.ts_ns,
                       lat_q=ping.lat_q,
                       lon_q=ping.lon_q,
                       event_code=ping.event_code,
                       hash_pref_len=len(ping.hash_pref)))
            dec += 1
    if dec == 0:
        fail("no decodable pings found in out/pings.bin", code=3)

    print("✅ Pipeline check completed successfully.")

if __name__ == "__main__":
    main()
