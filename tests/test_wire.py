from io import BytesIO

from src.wire import salute_pb2
from src.wire.codec import write_ld_stream, read_ld_stream


def test_codec_roundtrip_and_size():
    m = salute_pb2.SalutePing()
    m.uid = 7
    m.ts_ns = 1234567890
    m.lat_q = 374220000
    m.lon_q = -1220840000
    m.event_code = 2
    payload = m.SerializeToString()
    assert len(payload) <= 80

    buf = BytesIO()
    write_ld_stream([payload], buf)
    buf.seek(0)
    items = list(read_ld_stream(buf))
    assert len(items) == 1
    m2 = salute_pb2.SalutePing()
    m2.ParseFromString(items[0])
    assert m2.uid == m.uid
    assert m2.event_code == 2

