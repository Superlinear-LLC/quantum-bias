"""Review-round fixes: refill, stream depth, listener cleanup, per-mode stats, pool events, hosts."""
import asyncio
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_settings, new_session, prefetch_and_wait, wait_for
from quantum_lite.app import create_app
from quantum_lite.events import EventBus, sse_stream
from quantum_lite.pools import PoolBlock
from quantum_lite.providers.sim import SimProvider
from quantum_lite.stream_run import StreamRun

SESSION = {"provider": "sim", "mode": "buffered", "num_qubits": 5, "aggregation": "none",
           "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
           "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": None}


def blk(job, shots, n=5, at="2026-01-01T00:00:00+00:00"):
    return PoolBlock(job, "aer_ideal", tuple(range(n)), at, np.zeros((shots, n), dtype=np.uint8))


def stream_jobs(client):
    return [j for j in client.get("/api/jobs").json() if j["kind"] == "stream"]


def prefetch_jobs(client):
    return [j for j in client.get("/api/jobs").json() if j["kind"] == "prefetch"]


def wait_for_pool(client, key, shots):
    # A refill job is "done" before its bits are written to the pool (JobRunner.execute, then
    # PoolStore.append): wait for the shots, not the job state.
    wait_for(lambda: any(p["key"] == key and p["available_shots"] >= shots
                         for p in client.get("/api/pools").json()), what="the refill's shots in the pool")


# --- auto_refill --------------------------------------------------------------------------
def test_auto_refill_fires_when_pool_is_missing_but_the_request_still_409s(client):
    s = new_session(client, auto_refill=True, refill_below=50, shots_per_job=10)
    r = client.post(f"/api/sessions/{s['id']}/press")
    assert r.status_code == 409 and r.json()["error"] == "pool_empty"
    job = wait_for(lambda: next((j for j in prefetch_jobs(client) if j["state"] == "done"), None), what="refill")
    assert job["shots"] == 50 and job["session_id"] == s["id"]
    wait_for_pool(client, s["pool_key"], 10)
    assert client.post(f"/api/sessions/{s['id']}/press").status_code == 200


def test_auto_refill_fires_when_stream_start_finds_a_short_pool(client):
    mgr = client.app.state.mgr
    mgr.pools.append("sim", "aer_ideal", 5, blk("j0", 5))
    s = new_session(client, auto_refill=True, refill_below=1, shots_per_job=10)
    r = client.post(f"/api/sessions/{s['id']}/stream/start")
    assert r.status_code == 409 and r.json()["error"] == "pool_empty"
    wait_for(lambda: any(j["state"] == "done" for j in prefetch_jobs(client)), what="refill")
    wait_for_pool(client, s["pool_key"], 10)
    assert client.post(f"/api/sessions/{s['id']}/stream/start").status_code == 200


def test_auto_refill_fires_when_a_take_is_larger_than_the_remainder(client):
    mgr = client.app.state.mgr
    mgr.pools.append("sim", "aer_ideal", 5, blk("j0", 15))
    s = new_session(client, auto_refill=True, refill_below=1, shots_per_job=10)
    assert client.post(f"/api/sessions/{s['id']}/stream/start").status_code == 200
    # cycle 1 takes 10 (5 left, not below refill_below=1); cycle 2 is short: the stream dies
    # with pool_empty and that short take must still start the refill.
    job = wait_for(lambda: next((j for j in prefetch_jobs(client) if j["state"] == "done"), None), what="refill")
    assert job["session_id"] == s["id"]
    assert client.get(f"/api/sessions/{s['id']}").json()["stream"]["error"].startswith("pool_empty")


class FailingSim(SimProvider):
    """A simulator whose every job fails on the device; counts the jobs it was given."""

    def __init__(self, settings):
        super().__init__(settings)
        self.calls = 0

    def run(self, n, shots, on_state, stop):
        self.calls += 1
        raise RuntimeError("device error")


def test_failed_refill_is_not_retried_and_switches_auto_refill_off(tmp_path):
    settings = make_settings(tmp_path)
    sim = FailingSim(settings)
    with TestClient(create_app(settings, providers={"sim": sim})) as client:
        client.app.state.mgr.pools.append("sim", "aer_ideal", 5, blk("j0", 200))
        s = new_session(client, auto_refill=True, refill_below=190, shots_per_job=1)
        for _ in range(30):
            assert client.post(f"/api/sessions/{s['id']}/press").status_code == 200
        wait_for(lambda: not client.get(f"/api/sessions/{s['id']}").json()["auto_refill"], what="auto_refill off")
        for _ in range(5):
            client.post(f"/api/sessions/{s['id']}/press")
        assert sim.calls == 1 and len(prefetch_jobs(client)) == 1


# --- stream depth 1 / listener cleanup -------------------------------------------------------
def test_stream_restart_waits_for_the_in_flight_job(tmp_path):
    settings = make_settings(tmp_path, sim_latency_seconds=0.6)
    with TestClient(create_app(settings)) as c:
        s = new_session(c, mode="live")
        sid = s["id"]
        assert c.post(f"/api/sessions/{sid}/stream/start").status_code == 200
        wait_for(lambda: stream_jobs(c), what="the first run's job")
        assert c.post(f"/api/sessions/{sid}/stream/stop").status_code == 200
        assert stream_jobs(c)[0]["finished_at"] is None  # 0.6 s latency: still in flight
        r = c.post(f"/api/sessions/{sid}/stream/start")  # accepted: the new run's first cycle waits for it
        assert r.status_code == 200 and r.json()["stream"]["running"] is True, r.text
        assert len(stream_jobs(c)) == 1
        first, second = wait_for(lambda: len(j := stream_jobs(c)) >= 2 and j[:2], what="the new run's first job")
        assert first["finished_at"] is not None and first["discarded"] is True
        assert datetime.fromisoformat(second["submitted_at"]) >= datetime.fromisoformat(first["finished_at"])
        c.post(f"/api/sessions/{sid}/stream/stop")


def test_stream_stops_when_the_last_event_listener_disconnects(tmp_path):
    settings = make_settings(tmp_path, sim_latency_seconds=0.05)

    async def main():
        app = create_app(settings)
        mgr = app.state.mgr
        s = await mgr.create({**SESSION, "mode": "live"})
        first, second = mgr.bus.subscribe(s.id), mgr.bus.subscribe(s.id)
        await mgr.start_stream(s)
        mgr.bus.unsubscribe(first)
        mgr.listener_gone(s)
        assert s.streaming  # another listener remains
        mgr.bus.unsubscribe(second)
        mgr.listener_gone(s)
        await asyncio.sleep(0.2)
        assert not s.streaming and "listener" in s.stream_dict()["error"]
        await mgr.shutdown()

    asyncio.run(main())


def test_sse_stream_calls_on_close_when_the_client_goes():
    async def main():
        bus, closed = EventBus(), []
        sub = bus.subscribe("s1")
        gen = sse_stream(sub, bus, heartbeat=0.01, on_close=lambda: closed.append(bus.has_subscribers("s1")))
        await gen.__anext__()
        await gen.aclose()
        assert closed == [False]

    asyncio.run(main())


# --- stats per mode, provenance pieces, pool events --------------------------------------------
def test_stats_by_mode_combine_segments_of_the_same_mode(client):
    prefetch_and_wait(client, n=5, shots=100)
    s = new_session(client, mode="buffered")
    sid = s["id"]
    for mode in ("buffered", "live", "buffered", "live"):
        if client.get(f"/api/sessions/{sid}").json()["mode"] != mode:
            client.patch(f"/api/sessions/{sid}", json={"mode": mode})
        assert client.post(f"/api/sessions/{sid}/press").status_code == 200
    stats = client.get(f"/api/sessions/{sid}").json()["stats"]
    assert len(stats["segments"]) == 4
    assert stats["by_mode"]["buffered"]["shots"] == 2 and stats["by_mode"]["live"]["shots"] == 2
    assert stats["by_mode"]["live"]["raw"]["n"] == 10
    assert stats["total"]["shots"] == 4


def test_batch_spanning_blocks_reports_every_piece(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        mgr.pools.append("sim", "aer_ideal", 5, blk("j1", 5, at="2026-01-01T00:00:00+00:00"))
        mgr.pools.append("sim", "aer_ideal", 5, blk("j2", 5, at="2026-01-02T00:00:00+00:00"))
        s = await mgr.create({**SESSION, "shots_per_job": 8})
        cycle = await mgr.acq.cycle(s, StreamRun(60, None, mgr.clock))
        prov = cycle.batch.provenance
        assert prov["job_ids"] == ["j1", "j2"]
        assert [(p["job_id"], p["measured_at"][:10], p["shots"]) for p in prov["pieces"]] == \
            [("j1", "2026-01-01", 5), ("j2", "2026-01-02", 3)]

    asyncio.run(main())


def test_buffered_take_publishes_pool_event_with_current_counts(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        mgr.pools.append("sim", "aer_ideal", 5, blk("j1", 20))
        s = await mgr.create(SESSION)
        sub = mgr.bus.subscribe(None)
        await mgr.press(s)
        await mgr.acq.cycle(s, StreamRun(60, None, mgr.clock))
        pools = []
        while not sub.queue.empty():
            event, data = sub.queue.get_nowait()
            if event == "pool":
                pools.append(data["pool"])
        assert [p["consumed_shots"] for p in pools] == [1, 11]
        assert pools[-1]["available_shots"] == 9

    asyncio.run(main())


# --- host header, error shape, prefetch cap --------------------------------------------------------
def test_foreign_host_header_is_refused_on_local_bind(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings), base_url="http://rebound.evil.example") as c:
        r = c.post("/api/pools/prefetch", json={"provider": "sim", "num_qubits": 3, "shots": 10})
        assert r.status_code == 400
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8765") as c:
        assert c.get("/api/health").status_code == 200
    with TestClient(create_app(settings, bind_host="0.0.0.0"), base_url="http://192.168.1.5:8765") as c:
        assert c.get("/api/health").status_code == 200  # deliberate non-local bind: warned at start


def test_router_errors_use_the_error_detail_shape(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json() == {"error": "http_404", "detail": "Not Found"}
    r = client.delete("/api/health")
    assert r.status_code == 405 and r.json()["error"] == "http_405"


def test_sim_prefetch_size_is_capped(client):
    r = client.post("/api/pools/prefetch", json={"provider": "sim", "num_qubits": 3, "shots": 1_000_001})
    assert r.status_code == 400 and r.json()["error"] == "invalid_shots"
    assert client.get("/api/jobs").json() == []
