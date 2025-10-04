import time

from src.infra import store_forward
from src.infra.net_sim import set_online
from src.wire import salute_pb2


def test_offline_online_drain():
    store_forward.clear()
    set_online(False)

    for i in range(5):
        m = salute_pb2.SalutePing()
        m.uid = i + 1
        m.ts_ns = time.time_ns()
        store_forward.enqueue(m.SerializeToString(), ts_ns=m.ts_ns)

    assert store_forward.depth() == 5
    set_online(True)

    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        store_forward.drain(send_fn=lambda b: None, limit=100, budget_ms=50)
        if store_forward.depth() == 0:
            break
        time.sleep(0.05)
    assert store_forward.depth() == 0

