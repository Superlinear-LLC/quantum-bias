"""Live pipeline depth 1 across mode switches."""
import asyncio
import time
from datetime import datetime

import numpy as np

from conftest import make_settings, until
from quantum_lite.app import create_app
from quantum_lite.pools import PoolBlock
from test_event_payloads import GatedSim, http


def test_a_switch_to_live_with_a_stream_job_in_flight_waits_for_it(tmp_path):
    """The PATCH succeeds; the next live cycle waits for the stopped run's job, then submits its own.
    At no time are two stream jobs of the session in flight."""
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        app = create_app(settings, providers={"sim": sim})
        mgr = app.state.mgr
        mgr.pools.append("sim", "aer_ideal", 5, PoolBlock("j0", "aer_ideal", (0, 1, 2, 3, 4), "2026-01-01T00:00:00+00:00",
                                                          np.zeros((500, 5), dtype=np.uint8)))
        s = await mgr.create({"provider": "sim", "mode": "live", "num_qubits": 5, "aggregation": "none",
                              "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": None})

        def stream_jobs():
            return [r for r in mgr.runner.records.values() if r.kind == "stream"]

        try:
            async with http(app) as c:
                assert (await c.post(f"/api/sessions/{s.id}/stream/start")).status_code == 200
                await until(lambda: mgr.runner.has_inflight(s.id, "stream"), "the stream job")
                assert (await c.post(f"/api/sessions/{s.id}/stream/stop")).status_code == 200
                assert (await c.patch(f"/api/sessions/{s.id}", json={"mode": "buffered"})).status_code == 200
                assert (await c.post(f"/api/sessions/{s.id}/stream/start")).status_code == 200
                r = await c.patch(f"/api/sessions/{s.id}", json={"mode": "live"})
                assert r.status_code == 200 and r.json()["mode"] == "live", r.text
                await asyncio.sleep(0.3)  # many 20 ms cycles: the live one waits for the held job
                assert len(stream_jobs()) == 1 and s.streaming
                sim.gate.set()
                # Wait for two live batches recorded, not three jobs "done": a job can be done before its
                # cycle takes the result, and a stop in that gap rightly discards (and pools) it.
                await until(lambda: s.mode_stats["live"].snapshot()["shots"] >= 20, "two live batches recorded")
                assert (await c.post(f"/api/sessions/{s.id}/stream/stop")).status_code == 200
        finally:
            sim.gate.set()
        jobs = stream_jobs()
        assert jobs[0].discarded and not any(r.discarded for r in jobs[1:3])
        for earlier, later in zip(jobs, jobs[1:]):  # depth 1: each submitted after the one before finished
            assert earlier.finished_at is not None
            assert datetime.fromisoformat(later.submitted_at) >= datetime.fromisoformat(earlier.finished_at)
        assert s.mode_stats["live"].snapshot()["shots"] >= 20 and s.stream_dict()["error"] is None
        await mgr.shutdown()

    asyncio.run(main())


def test_a_switch_to_buffered_between_cycles_does_not_lose_pool_shots(tmp_path):
    # The stream loop used to decide "live" from s.mode, then let acquire read s.segment later.
    # A switch to buffered in between ran a pool take under the live give-up: at the deadline the
    # take was cancelled, its shots consumed but never recorded. Every shot taken from the pool
    # must be recorded (a buffered take running at the deadline completes).
    async def main():
        mgr = create_app(make_settings(tmp_path)).state.mgr
        s = await mgr.create({"provider": "sim", "mode": "live", "num_qubits": 5, "aggregation": "none",
                              "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 1, "stream_max_jobs": None})
        mgr.pools.append("sim", "aer_ideal", 5, PoolBlock("j0", "aer_ideal", (0, 1, 2, 3, 4), "2026-01-01T00:00:00+00:00",
                                                          np.zeros((100, 5), dtype=np.uint8)))
        loop = asyncio.get_running_loop()
        real_take, real_publish = mgr.pools.take, mgr.bus.publish
        mgr.pools.take = lambda key, shots: time.sleep(1.3) or real_take(key, shots)  # outlasts the 1 s deadline
        switched = []

        def publish(event, data, session_id=None):
            real_publish(event, data, session_id)
            if event == "stats" and session_id == s.id and not switched:  # end of the first (live) cycle
                switched.append(True)
                loop.call_soon(mgr.switch_mode, s, "buffered")  # lands before the next cycle's acquire runs

        mgr.bus.publish = publish
        await mgr.start_stream(s)
        await s.stream_task
        await asyncio.sleep(1.5)  # let a cancelled take's thread finish consuming
        consumed = mgr.pools.get(s.pool_key).cursor
        recorded = s.mode_stats["buffered"].snapshot()["shots"]
        assert consumed == recorded, (consumed, recorded, s.stream_dict())
        await mgr.shutdown()

    asyncio.run(main())


def test_a_switch_to_buffered_while_the_live_cycle_waits_submits_no_live_job(tmp_path):
    """A restarted live stream waits for the stopped run's job; a PATCH to buffered meanwhile is
    obeyed: the stream takes from the pool, and no stream job is submitted after the switch."""
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        app = create_app(settings, providers={"sim": sim})
        mgr = app.state.mgr
        mgr.pools.append("sim", "aer_ideal", 5, PoolBlock("j0", "aer_ideal", (0, 1, 2, 3, 4), "2026-01-01T00:00:00+00:00",
                                                          np.zeros((500, 5), dtype=np.uint8)))
        s = await mgr.create({"provider": "sim", "mode": "live", "num_qubits": 5, "aggregation": "none",
                              "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": None})

        def stream_jobs():
            return [r for r in mgr.runner.records.values() if r.kind == "stream"]

        try:
            async with http(app) as c:
                assert (await c.post(f"/api/sessions/{s.id}/stream/start")).status_code == 200
                await until(lambda: mgr.runner.has_inflight(s.id, "stream"), "the stream job")
                assert (await c.post(f"/api/sessions/{s.id}/stream/stop")).status_code == 200
                assert (await c.post(f"/api/sessions/{s.id}/stream/start")).status_code == 200
                await asyncio.sleep(0.1)  # the live cycle waits for the held job
                assert (await c.patch(f"/api/sessions/{s.id}", json={"mode": "buffered"})).status_code == 200
                sim.gate.set()
                await until(lambda: s.mode_stats["buffered"].snapshot()["shots"] >= 20, "buffered batches")
                assert (await c.post(f"/api/sessions/{s.id}/stream/stop")).status_code == 200
        finally:
            sim.gate.set()
        [held] = stream_jobs()  # no stream job after the switch
        assert held.discarded and s.mode_stats["live"].snapshot()["shots"] == 0
        assert s.segments[0].stats.snapshot()["shots"] == 0 and s.stream_dict()["error"] is None
        await mgr.shutdown()

    asyncio.run(main())
