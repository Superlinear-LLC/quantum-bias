"""Refill ceilings, pool integrity, IPv6 host, guard ownership, UI logic."""
import asyncio
import base64
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_settings, new_session
from fakes import FakeIbm
from quantum_lite.app import create_app
from quantum_lite.pools import PoolBlock, PoolStore
from quantum_lite.providers.ibm import IbmProvider
from quantum_lite.providers.sim import SimProvider

IBM_SESSION = {"provider": "ibm", "mode": "buffered", "num_qubits": 3, "aggregation": "none",
               "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 100,
               "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": 5}


def ibm_app(tmp_path, fake, **kw):
    settings = make_settings(tmp_path, ibm_token="TOK-123", **kw)
    prov = IbmProvider(settings, service_factory=fake.service_factory, sampler_cls=fake.sampler_cls,
                       poll_seconds=0.01)
    return create_app(settings, providers={"sim": SimProvider(settings), "ibm": prov})


def drain(sub):
    out = []
    while not sub.queue.empty():
        out.append(sub.queue.get_nowait())
    return out


# --- refill sizing (the ibm refill-ceiling cases are gone: auto_refill is sim-only) ----
def test_refill_smaller_than_threshold_is_400(client):
    r = client.post("/api/sessions", json={"provider": "sim", "auto_refill": True, "refill_below": 50,
                                           "refill_shots": 1, "stream_seconds": 60})
    assert r.status_code == 400 and r.json()["error"] == "invalid_refill"


# --- truncated pool block ----------------------------------------------------------
def test_truncated_pool_block_fails_at_load(tmp_path):
    store = PoolStore(tmp_path)
    store.append("sim", "aer_ideal", 5, PoolBlock("j", "aer_ideal", (0, 1, 2, 3, 4), "t", np.ones((1000, 5), np.uint8)))
    path = next((tmp_path / "pools").glob("*.json"))
    data = json.loads(path.read_text())
    raw = base64.b64decode(data["blocks"][0]["bits"])
    data["blocks"][0]["bits"] = base64.b64encode(raw[: len(raw) // 2]).decode()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="bytes of bits"):
        PoolStore(tmp_path)


# --- IPv6 bind ------------------------------------------------------------------------
def test_ipv6_loopback_bind_is_non_local_and_serves(tmp_path, monkeypatch, capsys):
    from quantum_lite import app as app_mod
    from quantum_lite import cli

    with TestClient(create_app(make_settings(tmp_path), bind_host="::1"), base_url="http://[::1]:8765") as c:
        assert c.get("/api/health").status_code == 200
    import uvicorn

    seen = {}
    real = cli.make_server

    def spy(app, host, port, open_url=None):
        seen["open_url"] = open_url
        return real(app, host, port, open_url)

    monkeypatch.setattr(cli, "make_server", spy)
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, *a, **k: None)
    monkeypatch.setenv("QL_DATA_DIR", str(tmp_path / "d"))
    assert cli.main(["--host", "::1", "--open"]) == 0
    assert "WARNING" in capsys.readouterr().err
    assert seen["open_url"] == "http://[::1]:8765/"


# --- a job is not 'queued' before the provider says so --------------------------------
def test_ibm_job_starts_submitting_then_queued(tmp_path):
    fake = FakeIbm()
    app = ibm_app(tmp_path, fake)

    async def main():
        mgr = app.state.mgr
        s = await mgr.create({**IBM_SESSION, "mode": "live", "auto_refill": False})
        sub = mgr.bus.subscribe(s.id)
        await mgr.press(s)
        jobs = [d for e, d in drain(sub) if e == "job"]
        assert (jobs[0]["state"], jobs[0]["job_id"]) == ("submitting", None)
        assert jobs[1]["state"] == "queued" and jobs[1]["job_id"] == "fakejob-1"

    asyncio.run(main())


# --- a manual prefetch must not clear a refill's guard -------------------------------
class GatedSim(SimProvider):
    """A job of `gated_shots` shots (the refill) waits for `gate`; other jobs run at once.

    Gated by which job it is (its shot count), not by call order or latency.
    """

    def __init__(self, settings, gated_shots):
        super().__init__(settings)
        self.gate = threading.Event()
        self.gated_shots = gated_shots

    def run(self, n, shots, on_state, stop):
        if shots == self.gated_shots:
            assert self.gate.wait(10), "the test never opened the gate"
        return super().run(n, shots, on_state, stop)


async def landed(sub, job_id, timeout=10.0):
    """Wait, bounded, for the global `pool` event of `job_id`; fail at once on its `error` event."""
    deadline = time.monotonic() + timeout
    while True:
        for e, d in drain(sub):
            if d.get("job_id") == job_id:
                if e == "pool":
                    return d
                assert e != "error", f"job {job_id} failed to land: {d}"
        assert time.monotonic() < deadline, f"timed out waiting for job {job_id} to land"
        await asyncio.sleep(0.01)


def test_manual_prefetch_leaves_the_refill_guard_alone(tmp_path):
    from quantum_lite.pool_feed import TopUp

    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings, gated_shots=10)  # refill_shots below; the prefetch takes 20
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        feeder = mgr.feeder
        s = await mgr.create({"provider": "sim", "mode": "buffered", "num_qubits": 5, "aggregation": "none",
                              "shots_per_job": 1, "interval_ms": 20, "auto_refill": True, "refill_below": 10,
                              "refill_shots": 10, "stream_seconds": 60, "stream_max_jobs": None})
        sub = mgr.bus.subscribe(None)
        try:
            assert await feeder.top_up(s, force=True) is TopUp.SUBMITTED  # the refill: held by the gate
            placed = await feeder.prefetch("sim", 5, 20)
            assert placed.pool_key == s.pool_key == "sim__aer_ideal__5"
            # The prefetch lands while the refill is still held (drain would wait for the gate too).
            await landed(sub, placed.record.id)
            assert await feeder.top_up(s, force=True) is TopUp.BUSY
            assert len(mgr.runner.records) == 2
        finally:
            sim.gate.set()
        await feeder.drain()
        assert await feeder.top_up(s, force=True) is TopUp.SUBMITTED  # the refill's landing released it
        await feeder.drain()
        await mgr.shutdown()

    asyncio.run(main())


# --- pool file I/O runs off the event loop -----------------------------------------
def test_pool_take_runs_in_a_worker_thread(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        mgr.pools.append("sim", "aer_ideal", 5, PoolBlock("j", "aer_ideal", (0, 1, 2, 3, 4), "t", np.zeros((10, 5), np.uint8)))
        threads = []
        real = mgr.pools.take
        mgr.pools.take = lambda *a: threads.append(threading.current_thread()) or real(*a)
        s = await mgr.create({"provider": "sim", "mode": "buffered", "num_qubits": 5, "aggregation": "none",
                              "shots_per_job": 1, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": None})
        await mgr.press(s)
        assert threads and threads[0] is not threading.main_thread()

    asyncio.run(main())
