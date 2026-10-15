"""Acquisition seen from outside: provenance, event order and the error codes and texts, as literals.

Pinned before acquisition moved out of sessions.py; they pass unchanged after it. Everything is
driven through HTTP or the session manager's public methods, never its internals. Bits come from
the provider (sim, a failing or gated sim, ibm over FakeIbm) or from a seeded pool.
"""
import asyncio

import httpx
import numpy as np

from conftest import ibm_app, make_settings, until
from fakes import FakeIbm
from quantum_lite.app import create_app
from quantum_lite.pools import PoolBlock
from test_event_payloads import SIM_JOB, TS, Events, FailingSim, GatedSim, Like, Number, http, job, pool, same

LIVE = {"provider": "sim", "mode": "live", "num_qubits": 5, "aggregation": "none", "shots_per_job": 10,
        "interval_ms": 20, "auto_refill": False, "refill_below": 0, "refill_shots": None,
        "stream_seconds": 60, "stream_max_jobs": None}
BUFFERED = {**LIVE, "mode": "buffered"}
QUBITS = [0, 1, 2, 3, 4]
DAY1, DAY2 = "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"


def stream(**kw) -> dict:
    """A `stream` dict (Session.stream_dict) of a running run."""
    base = {"running": True, "error": None, "stopped_reason": None, "seconds_limit": 60,
            "seconds_left": Number(), "jobs_limit": None, "jobs_used": 0}
    base.update(kw)
    return base


def add(mgr, job_id, bits, at=DAY1):
    mgr.pools.append("sim", "aer_ideal", 5, PoolBlock(job_id, "aer_ideal", (0, 1, 2, 3, 4), at,
                                                      np.asarray(bits, dtype=np.uint8)))


def http_500(app):
    """Like http(), but an unhandled exception comes back as the app's 500 response instead of raising."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                             base_url="http://testserver")


def error(code, detail) -> dict:
    return {"error": code, "detail": detail}


async def stream_ended(s, timeout=10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while s.streaming:
        assert asyncio.get_running_loop().time() < deadline, "the stream did not end"
        await asyncio.sleep(0.01)


# --- provenance and event order -----------------------------------------------------------------
def test_live_press_body_provenance_and_event_order(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        s = await mgr.create(LIVE)
        ev, glob = Events(mgr.bus.subscribe(s.id)), Events(mgr.bus.subscribe(None))
        async with http(app) as c:
            r = await c.post(f"/api/sessions/{s.id}/press")
        assert r.status_code == 200, r.text
        body = r.json()
        assert list(body) == ["bit", "provenance", "stats"] and body["bit"] in (0, 1)
        prov = body["provenance"]
        same(prov, {"job_id": Like(SIM_JOB), "job_ids": [prov["job_id"]], "source": "live", "provider": "sim",
                    "backend": "aer_ideal", "measured_at": Like(TS), "physical_qubits": QUBITS,
                    "pieces": [{"job_id": prov["job_id"], "backend": "aer_ideal", "measured_at": prov["measured_at"],
                                "physical_qubits": QUBITS, "shots": 1}],
                    "segment": 0, "shots": 1})
        assert [(e, d.get("state")) for e, d in ev.pump()] == [
            ("job", "submitting"), ("job", "queued"), ("job", "running"), ("job", "done"), ("bit", None), ("stats", None)]
        same(ev.all[0][1], {**job(kind="press", shots=1, n_qubits=5, session_id=s.id), "ts": Like(TS)})
        same(ev.all[4][1], {"kind": "press", "bit": body["bit"], "provenance": prov, "ts": Like(TS), "session_id": s.id})
        assert list(ev.all[5][1]) == ["stats", "ts", "session_id"] and ev.all[5][1]["stats"] == body["stats"]
        assert [e for e, _ in glob.pump()] == ["job"] * 4  # a live press touches no pool
        await mgr.shutdown()

    asyncio.run(main())


def test_buffered_batch_spanning_two_blocks_event_literals(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        add(mgr, "j1", np.eye(5), DAY1)  # column 0: 1 0 0 0 0; 5 ones
        add(mgr, "j2", np.ones((5, 5)), DAY2)  # the batch takes 3 rows: 15 ones
        s = await mgr.create({**BUFFERED, "shots_per_job": 8, "interval_ms": 5000})
        ev, glob = Events(mgr.bus.subscribe(s.id)), Events(mgr.bus.subscribe(None))
        await mgr.start_stream(s)
        await ev.until(lambda: ev.of("stats"), "the first cycle's stats")
        await mgr.stop_stream(s)
        assert [e for e, _ in ev.pump()] == ["stream", "batch", "stats"]
        same(ev.all[0][1], {"stream": stream(), "ts": Like(TS), "session_id": s.id})
        same(ev.all[1][1], {
            "shots": 8, "raw_ones": 20, "aggregated_ones": 4, "tail_bit": 1, "tail_run": 3,
            "provenance": {
                "job_id": "j1", "job_ids": ["j1", "j2"], "source": "buffered", "provider": "sim", "backend": "aer_ideal",
                "measured_at": DAY1, "physical_qubits": QUBITS,
                "pieces": [{"job_id": "j1", "backend": "aer_ideal", "measured_at": DAY1, "physical_qubits": QUBITS, "shots": 5},
                           {"job_id": "j2", "backend": "aer_ideal", "measured_at": DAY2, "physical_qubits": QUBITS, "shots": 3}],
                "segment": 0, "shots": 8},
            "ts": Like(TS), "session_id": s.id})
        assert list(ev.all[2][1]) == ["stats", "ts", "session_id"]
        assert ev.all[2][1]["stats"]["total"]["shots"] == 8
        assert [e for e, _ in glob.pump()] == ["pool"]
        same(glob.all[0][1], {"pool": pool("sim__aer_ideal__5", 5, [
            {"job_id": "j1", "backend": "aer_ideal", "shots": 5, "measured_at": DAY1},
            {"job_id": "j2", "backend": "aer_ideal", "shots": 5, "measured_at": DAY2}], consumed=8), "ts": Like(TS)})
        await mgr.shutdown()

    asyncio.run(main())


def test_live_stream_run_of_one_job_event_order_and_literals(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        s = await mgr.create({**LIVE, "stream_max_jobs": 1})
        ev = Events(mgr.bus.subscribe(s.id))
        await mgr.start_stream(s)
        await stream_ended(s)
        assert [(e, d.get("state")) for e, d in ev.pump()] == [
            ("stream", None), ("job", "submitting"), ("stream", None), ("job", "queued"), ("job", "running"),
            ("job", "done"), ("batch", None), ("stats", None), ("stream_stopped", None)]
        same(ev.all[0][1], {"stream": stream(jobs_limit=1), "ts": Like(TS), "session_id": s.id})
        same(ev.all[1][1], {**job(kind="stream", shots=10, n_qubits=5, session_id=s.id), "ts": Like(TS)})
        same(ev.all[2][1], {"stream": stream(jobs_limit=1, jobs_used=1), "ts": Like(TS), "session_id": s.id})
        rid, jid = ev.all[1][1]["id"], ev.all[5][1]["job_id"]
        assert ev.all[5][1]["id"] == rid
        same(ev.all[6][1], {
            "shots": 10, "raw_ones": Number(), "aggregated_ones": Number(), "tail_bit": Number(), "tail_run": Number(),
            "provenance": {
                "job_id": Like(SIM_JOB), "job_ids": [jid], "source": "live", "provider": "sim", "backend": "aer_ideal",
                "measured_at": Like(TS), "physical_qubits": QUBITS,
                "pieces": [{"job_id": jid, "backend": "aer_ideal", "measured_at": Like(TS), "physical_qubits": QUBITS,
                            "shots": 10}],
                "segment": 0, "shots": 10},
            "ts": Like(TS), "session_id": s.id})
        detail = "1 jobs caused by this stream, the limit is 1"
        same(ev.all[8][1], {"code": "limit_jobs", "detail": detail,
                            "stream": stream(running=False, stopped_reason=f"limit_jobs: {detail}", seconds_left=None,
                                             jobs_limit=1, jobs_used=1),
                            "ts": Like(TS), "session_id": s.id})
        await mgr.shutdown()

    asyncio.run(main())


def test_a_switch_while_a_live_press_is_in_flight_leaves_its_bit_live_in_segment_0(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        app = create_app(settings, providers={"sim": sim})
        mgr = app.state.mgr
        s = await mgr.create(LIVE)
        ev = Events(mgr.bus.subscribe(s.id))
        try:
            async with http(app) as c:
                press = asyncio.ensure_future(c.post(f"/api/sessions/{s.id}/press"))
                await ev.until(lambda: mgr.runner.has_inflight(s.id, "press"), "the press job")
                r = await c.patch(f"/api/sessions/{s.id}", json={"mode": "buffered"})
                assert r.status_code == 200 and r.json()["mode"] == "buffered", r.text
                sim.gate.set()
                r = await press
        finally:
            sim.gate.set()
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provenance"]["source"] == "live" and body["provenance"]["segment"] == 0
        segments = body["stats"]["segments"]
        assert [(g["index"], g["mode"], g["stats"]["shots"]) for g in segments] == [(0, "live", 1), (1, "buffered", 0)]
        assert body["stats"]["by_mode"]["live"]["shots"] == 1 and body["stats"]["by_mode"]["buffered"]["shots"] == 0
        await mgr.shutdown()

    asyncio.run(main())


# --- failures: 502, 500 and the stream's error event --------------------------------------------
def test_a_failing_live_press_is_502_with_the_real_text_and_touches_no_pool(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        app = create_app(settings, providers={"sim": FailingSim(settings)})
        mgr = app.state.mgr
        s = await mgr.create(LIVE)
        ev = Events(mgr.bus.subscribe(s.id))
        async with http(app) as c:
            r = await c.post(f"/api/sessions/{s.id}/press")
        assert r.status_code == 502
        same(r.json(), error("provider_error", "RuntimeError: device error"))
        assert [(e, d.get("state")) for e, d in ev.pump()] == [("job", "submitting"), ("job", "error")]
        assert mgr.pools.list() == [] and s.stats.snapshot()["shots"] == 0
        await mgr.shutdown()

    asyncio.run(main())


def test_a_failing_live_stream_job_is_a_provider_error_event(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        app = create_app(settings, providers={"sim": FailingSim(settings)})
        mgr = app.state.mgr
        s = await mgr.create(LIVE)
        ev = Events(mgr.bus.subscribe(s.id))
        await mgr.start_stream(s)
        await stream_ended(s)
        assert [(e, d.get("state")) for e, d in ev.pump()] == [
            ("stream", None), ("job", "submitting"), ("stream", None), ("job", "error"), ("error", None)]
        same(ev.all[4][1], {"scope": "stream", "error": "provider_error", "detail": "RuntimeError: device error",
                            "ts": Like(TS), "session_id": s.id})
        same(s.stream_dict(), stream(running=False, error="provider_error: RuntimeError: device error",
                                     seconds_left=None, jobs_used=1))
        assert mgr.pools.list() == []
        await mgr.shutdown()

    asyncio.run(main())


def test_a_store_fault_on_a_buffered_press_is_500_not_502(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        s = await mgr.create(BUFFERED)

        def take(key, shots):
            raise OSError("disk full")

        mgr.pools.take = take
        async with http_500(app) as c:
            r = await c.post(f"/api/sessions/{s.id}/press")
        assert r.status_code == 500
        same(r.json(), error("internal_error", "OSError: disk full"))
        await mgr.shutdown()

    asyncio.run(main())


def test_a_store_fault_in_a_buffered_stream_ends_it_as_stream_error(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        add(mgr, "j1", np.zeros((50, 5)))
        s = await mgr.create(BUFFERED)

        def take(key, shots):
            raise OSError("disk full")

        mgr.pools.take = take
        ev = Events(mgr.bus.subscribe(s.id))
        await mgr.start_stream(s)
        await stream_ended(s)
        assert [e for e, _ in ev.pump()] == ["stream", "error"]
        same(ev.all[1][1], {"scope": "stream", "error": "stream_error", "detail": "disk full",
                            "ts": Like(TS), "session_id": s.id})
        assert s.stream_dict()["error"] == "stream_error: disk full"
        await mgr.shutdown()

    asyncio.run(main())


def test_ibm_job_error_on_press_is_502_with_the_job_and_message(tmp_path):
    async def main():
        fake = FakeIbm(statuses=("ERROR",), error="quota exhausted")
        app = ibm_app(tmp_path, fake)
        mgr = app.state.mgr
        s = await mgr.create({**LIVE, "provider": "ibm", "num_qubits": 3, "stream_max_jobs": 5})
        async with http(app) as c:
            r = await c.post(f"/api/sessions/{s.id}/press")
        assert r.status_code == 502
        same(r.json(), error("provider_error", "ProviderError: IBM job fakejob-1 ended error: quota exhausted"))
        assert fake.jobs == 1 and mgr.pools.list() == []
        await mgr.shutdown()

    asyncio.run(main())


# --- refusals: 400, 409, 429 --------------------------------------------------------------------
def test_pool_empty_texts_for_a_press_and_a_stream_start(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        s = await mgr.create(BUFFERED)
        async with http(app) as c:
            r = await c.post(f"/api/sessions/{s.id}/press")
            assert r.status_code == 409
            same(r.json(), error("pool_empty", "pool sim__aer_ideal__5 does not exist (0 unconsumed shots, 1 requested)"))
            add(mgr, "j1", np.zeros((5, 5)))
            r = await c.post(f"/api/sessions/{s.id}/stream/start")
            assert r.status_code == 409
            same(r.json(), error("pool_empty", "pool sim__aer_ideal__5 has 5 unconsumed shots, stream needs 10 per cycle"))
            for _ in range(5):
                assert (await c.post(f"/api/sessions/{s.id}/press")).status_code == 200
            r = await c.post(f"/api/sessions/{s.id}/predict", json={"guess": 1})
            assert r.status_code == 409
            same(r.json(), error("pool_empty", "pool sim__aer_ideal__5 has 0 unconsumed shots, 1 requested"))
            assert mgr.runner.records == {}  # auto_refill is off: nothing was submitted
        await mgr.shutdown()

    asyncio.run(main())


def test_a_buffered_stream_that_runs_dry_is_a_pool_empty_error_event(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        add(mgr, "j1", np.zeros((15, 5)))
        s = await mgr.create(BUFFERED)
        ev = Events(mgr.bus.subscribe(s.id))
        await mgr.start_stream(s)
        await stream_ended(s)
        assert [e for e, _ in ev.pump()] == ["stream", "batch", "stats", "error"]
        detail = "pool sim__aer_ideal__5 has 5 unconsumed shots, 10 requested"
        same(ev.all[3][1], {"scope": "stream", "error": "pool_empty", "detail": detail, "ts": Like(TS), "session_id": s.id})
        assert s.stream_dict()["error"] == f"pool_empty: {detail}" and s.stream_dict()["stopped_reason"] is None
        await mgr.shutdown()

    asyncio.run(main())


def test_stream_running_and_perceive_pending_texts(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        add(mgr, "j1", np.zeros((500, 5)))
        s = await mgr.create({**BUFFERED, "interval_ms": 1000})
        async with http(app) as c:
            assert (await c.post(f"/api/sessions/{s.id}/stream/start")).status_code == 200
            r = await c.post(f"/api/sessions/{s.id}/stream/start")
            assert r.status_code == 409
            same(r.json(), error("stream_running", "stream already running"))
            assert (await c.post(f"/api/sessions/{s.id}/stream/stop")).status_code == 200
            assert (await c.post(f"/api/sessions/{s.id}/perceive/prepare")).status_code == 200
            r = await c.post(f"/api/sessions/{s.id}/perceive/prepare")
            assert r.status_code == 409
            same(r.json(), error("perceive_pending", "a perceive trial is already prepared; reveal it first"))
        await mgr.shutdown()

    asyncio.run(main())


def test_switch_400s_and_a_start_or_switch_to_live_with_a_stream_job_in_flight(tmp_path):
    """Depth 1 is kept at the permit, not by refusing: with a stopped run's job still in flight a live
    stream start and a PATCH to live succeed, and no second stream job is submitted while it runs."""
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        app = create_app(settings, providers={"sim": sim})
        mgr = app.state.mgr
        add(mgr, "j1", np.zeros((500, 5)))
        s = await mgr.create({**LIVE, "interval_ms": 1000})

        def stream_jobs():
            return [r for r in mgr.runner.records.values() if r.kind == "stream"]

        try:
            async with http(app) as c:
                sid = s.id
                assert (await c.post(f"/api/sessions/{sid}/stream/start")).status_code == 200
                await until(lambda: mgr.runner.has_inflight(sid, "stream"), "the stream job")
                assert (await c.post(f"/api/sessions/{sid}/stream/stop")).status_code == 200
                r = await c.post(f"/api/sessions/{sid}/stream/start")  # live, the stopped run's job still held
                assert r.status_code == 200, r.text
                same(r.json()["stream"], stream())
                await asyncio.sleep(0.2)
                assert len(stream_jobs()) == 1 and s.streaming  # the live cycle waits for the held job
                assert (await c.post(f"/api/sessions/{sid}/stream/stop")).status_code == 200
                assert (await c.patch(f"/api/sessions/{sid}", json={"mode": "buffered"})).status_code == 200
                assert (await c.post(f"/api/sessions/{sid}/stream/start")).status_code == 200
                r = await c.patch(f"/api/sessions/{sid}", json={"mode": "x"})
                assert r.status_code == 400
                same(r.json(), error("invalid_mode", "mode must be one of ('live', 'buffered'), got 'x'"))
                r = await c.patch(f"/api/sessions/{sid}", json={"mode": "buffered"})
                assert r.status_code == 400
                same(r.json(), error("mode_unchanged", "session is already buffered"))
                r = await c.patch(f"/api/sessions/{sid}", json={"mode": "live"})
                assert r.status_code == 200 and r.json()["mode"] == "live", r.text
                assert [g.mode for g in s.segments] == ["live", "buffered", "live"]
                await asyncio.sleep(1.2)  # past the buffered cycle's 1 s interval: the next cycle is live
                assert len(stream_jobs()) == 1 and s.streaming and s.stream_dict()["error"] is None
                assert (await c.post(f"/api/sessions/{sid}/stream/stop")).status_code == 200
        finally:
            sim.gate.set()
        await mgr.shutdown()

    asyncio.run(main())


def test_ibm_job_ceiling_is_429_on_a_press_and_a_job_ceiling_error_in_a_stream(tmp_path):
    async def main():
        fake = FakeIbm()
        app = ibm_app(tmp_path, fake, ibm_max_jobs=1)
        mgr = app.state.mgr
        body = {**LIVE, "provider": "ibm", "num_qubits": 3, "stream_max_jobs": 5}
        s = await mgr.create(body)
        detail = "IBM_MAX_JOBS is 1, 1 already submitted by this process"
        async with http(app) as c:
            assert (await c.post(f"/api/sessions/{s.id}/press")).status_code == 200
            r = await c.post(f"/api/sessions/{s.id}/press")
            assert r.status_code == 429
            same(r.json(), error("job_ceiling", detail))
        assert fake.jobs == 1 and len(mgr.runner.records) == 1
        t = await mgr.create(body)
        ev = Events(mgr.bus.subscribe(t.id))
        await mgr.start_stream(t)
        await stream_ended(t)
        assert [e for e, _ in ev.pump()] == ["stream", "error"]  # refused at the submit: no job, nothing counted
        same(ev.all[1][1], {"scope": "stream", "error": "job_ceiling", "detail": detail, "ts": Like(TS), "session_id": t.id})
        same(t.stream_dict(), stream(running=False, error=f"job_ceiling: {detail}", seconds_left=None, jobs_limit=5))
        assert fake.jobs == 1
        await mgr.shutdown()

    asyncio.run(main())
