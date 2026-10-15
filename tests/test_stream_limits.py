"""A stream is bounded by the server: by wall time (stream_seconds) and, live, by jobs (stream_max_jobs).

The browser is not involved anywhere here: nothing but the server's own loop stops these streams.
"""
import asyncio
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from conftest import ibm_app, make_settings, new_session, prefetch_and_wait, until, wait_for
from fakes import FakeIbm
from quantum_lite.acquisition import Cycle, Stop
from quantum_lite.app import create_app
from quantum_lite.stream_run import StreamRun

def stream_of(c, sid):
    return c.get(f"/api/sessions/{sid}").json()["stream"]


def parse_ts(s):
    return datetime.fromisoformat(s).astimezone(timezone.utc).timestamp()


# --- time limit ------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["live", "buffered"])
def test_stream_stops_by_itself_at_stream_seconds(tmp_path, mode):
    with TestClient(create_app(make_settings(tmp_path, sim_latency_seconds=0.05))) as c:
        if mode == "buffered":
            prefetch_and_wait(c, n=5, shots=3000)
        s = new_session(c, mode=mode, shots_per_job=10, interval_ms=20, stream_seconds=1)
        sid = s["id"]
        assert s["stream_seconds"] == 1 and s["stream"]["seconds_limit"] == 1
        t0 = time.time()
        r = c.post(f"/api/sessions/{sid}/stream/start")
        assert r.status_code == 200, r.text
        started = r.json()["stream"]
        t_resp = time.time()
        assert started["running"] is True and 0 < started["seconds_left"] <= 1
        wait_for(lambda: not stream_of(c, sid)["running"], timeout=6, what="the stream to stop by itself")
        stopped_after = time.time() - t0
        st = stream_of(c, sid)
        assert 0.9 <= stopped_after < 2.5, stopped_after
        assert st["stopped_reason"].startswith("limit_seconds:") and st["error"] is None
        assert st["seconds_left"] is None
        shots = c.get(f"/api/sessions/{sid}").json()["stats"]["total"]["shots"]
        assert shots >= 30  # it really streamed before the limit
        # no cycle began after the deadline (every job was submitted before start + 1 s)
        stream_jobs = [j for j in c.get("/api/jobs").json() if j["kind"] == "stream"]
        if mode == "live":
            assert stream_jobs and all(parse_ts(j["submitted_at"]) <= t_resp + 1.0 for j in stream_jobs)
        time.sleep(0.5)
        assert c.get(f"/api/sessions/{sid}").json()["stats"]["total"]["shots"] == shots  # nothing after the stop
        assert len([j for j in c.get("/api/jobs").json() if j["kind"] == "stream"]) == len(stream_jobs)
        # stopping an already stopped stream is the usual 409; a new run clears the reason
        assert c.post(f"/api/sessions/{sid}/stream/stop").status_code == 409


def test_buffered_stream_does_not_sleep_past_the_deadline(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        prefetch_and_wait(c, n=5, shots=1000)
        s = new_session(c, mode="buffered", shots_per_job=10, interval_ms=5000, stream_seconds=1)
        sid = s["id"]
        t0 = time.time()
        assert c.post(f"/api/sessions/{sid}/stream/start").status_code == 200
        wait_for(lambda: not stream_of(c, sid)["running"], timeout=8, what="the stream to stop at the deadline")
        assert time.time() - t0 < 2.5  # not after the 5 s interval sleep
        st = stream_of(c, sid)
        assert (st["stopped_reason"] or "").startswith("limit_seconds:"), st


def test_job_in_flight_at_the_deadline_is_left_to_finish_and_pooled(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, sim_latency_seconds=2.5))) as c:
        s = new_session(c, mode="live", shots_per_job=10, stream_seconds=1)
        sid = s["id"]
        t0 = time.time()
        assert c.post(f"/api/sessions/{sid}/stream/start").status_code == 200
        wait_for(lambda: not stream_of(c, sid)["running"], timeout=5, what="the stream to stop at the deadline")
        assert time.time() - t0 < 2.0  # stopped at 1 s, not when the 2.5 s job ended
        st = stream_of(c, sid)
        assert st["stopped_reason"].startswith("limit_seconds:") and "in flight" in st["stopped_reason"]
        jobs = [j for j in c.get("/api/jobs").json() if j["kind"] == "stream"]
        assert len(jobs) == 1 and jobs[0]["state"] != "done"
        # it cannot be recalled: it finishes, is marked discarded and its bits go to the pool
        job = wait_for(lambda: next((j for j in c.get("/api/jobs").json() if j["kind"] == "stream" and j["pooled_to"]), None),
                       timeout=8, what="the discarded job to be pooled")
        assert job["discarded"] is True and job["state"] == "done"
        assert c.get(f"/api/sessions/{sid}").json()["stats"]["total"]["shots"] == 0  # not counted as live bits
        # the job is done now, so the same session can stream again
        assert c.post(f"/api/sessions/{sid}/stream/start").status_code == 200
        assert stream_of(c, sid)["stopped_reason"] is None
        c.post(f"/api/sessions/{sid}/stream/stop")


# --- job limit -------------------------------------------------------------------------------
def test_ibm_live_stream_stops_after_exactly_stream_max_jobs(tmp_path):
    fake = FakeIbm()
    with TestClient(ibm_app(tmp_path, fake)) as c:
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=10, stream_seconds=60,
                        stream_max_jobs=2)
        sid = s["id"]
        assert s["stream_max_jobs"] == 2 and s["stream"]["jobs_limit"] == 2
        assert c.post(f"/api/sessions/{sid}/stream/start").status_code == 200
        wait_for(lambda: not stream_of(c, sid)["running"], timeout=10, what="the stream to stop at its job limit")
        time.sleep(0.3)  # a third job would have been submitted by now
        st = stream_of(c, sid)
        assert st["stopped_reason"].startswith("limit_jobs:") and st["error"] is None
        assert st["jobs_used"] == 2 and st["seconds_left"] is None
        assert fake.jobs == 2
        assert len([j for j in c.get("/api/jobs").json() if j["kind"] == "stream"]) == 2
        assert c.get("/api/budget").json()["jobs"]["used"] == 2
        assert c.get(f"/api/sessions/{sid}").json()["stats"]["total"]["shots"] == 20  # both jobs were counted


def test_job_limit_counts_this_stream_run_not_the_process(tmp_path):
    fake = FakeIbm()
    with TestClient(ibm_app(tmp_path, fake)) as c:
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=10, stream_max_jobs=1)
        sid = s["id"]
        for run in (1, 2):  # two runs of one job each: the process-wide count (2) is not the run count
            assert c.post(f"/api/sessions/{sid}/stream/start").status_code == 200
            wait_for(lambda: not stream_of(c, sid)["running"], timeout=10, what=f"run {run} to stop")
            assert stream_of(c, sid)["jobs_used"] == 1
        assert fake.jobs == 2


def test_job_limit_applies_when_the_mode_is_patched_to_live_later(tmp_path):
    fake = FakeIbm()
    with TestClient(ibm_app(tmp_path, fake)) as c:
        s = new_session(c, provider="ibm", mode="buffered", num_qubits=3, shots_per_job=10, stream_max_jobs=2)
        sid = s["id"]
        c.patch(f"/api/sessions/{sid}", json={"mode": "live"})
        assert c.post(f"/api/sessions/{sid}/stream/start").status_code == 200
        wait_for(lambda: not stream_of(c, sid)["running"], timeout=10, what="the stream to stop")
        assert fake.jobs == 2 and stream_of(c, sid)["stopped_reason"].startswith("limit_jobs:")


def test_limit_stops_are_published_as_stream_stopped_events(tmp_path):
    fake = FakeIbm()

    async def main():
        app = ibm_app(tmp_path, fake)
        mgr = app.state.mgr
        s = await mgr.create({"provider": "ibm", "mode": "live", "num_qubits": 3, "aggregation": "none",
                              "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": 2})
        sub = mgr.bus.subscribe(s.id)
        await mgr.start_stream(s)
        while s.streaming:
            await asyncio.sleep(0.02)
        got = []
        while not sub.queue.empty():
            got.append(sub.queue.get_nowait())
        stopped = [d for e, d in got if e == "stream_stopped"]
        assert len(stopped) == 1 and stopped[0]["code"] == "limit_jobs"
        assert stopped[0]["stream"]["running"] is False and stopped[0]["stream"]["jobs_used"] == 2
        assert stopped[0]["stream"]["stopped_reason"].startswith("limit_jobs:")
        assert not [e for e, d in got if e == "error"]  # a limit is not an error
        progress = [d["stream"]["jobs_used"] for e, d in got if e == "stream"]
        assert progress == [0, 1, 2]  # start, then one per submitted job
        await mgr.shutdown()

    asyncio.run(main())


# --- create validation -----------------------------------------------------------------------
def create(c, **kw):
    body = {"provider": "sim", "mode": "buffered", "stream_seconds": 30}
    body.update(kw)
    return c.post("/api/sessions", json={k: v for k, v in body.items() if v != "omit"})


def test_create_without_stream_seconds_is_400(client):
    r = create(client, stream_seconds="omit")
    assert r.status_code == 400 and r.json()["error"] == "invalid_stream_seconds"
    assert "stream_seconds is required" in r.json()["detail"]
    r = create(client, stream_seconds=None)
    assert r.status_code == 400 and r.json()["error"] == "invalid_stream_seconds"


@pytest.mark.parametrize("value", [0, -5, 3601])
def test_stream_seconds_out_of_range_is_400_with_the_reason(client, value):
    r = create(client, stream_seconds=value)
    assert r.status_code == 400 and r.json()["error"] == "invalid_stream_seconds"
    assert f"got {value}" in r.json()["detail"] and "3600" in r.json()["detail"]


@pytest.mark.parametrize("value", [True, 2.5, "30", [30]])
def test_stream_seconds_must_be_an_integer(client, value):
    r = create(client, stream_seconds=value)
    assert r.status_code == 400 and r.json()["error"] == "validation_error"
    assert "stream_seconds" in r.json()["detail"]


def test_stream_seconds_bounds_are_accepted(client):
    for v in (1, 3600):
        r = create(client, stream_seconds=v)
        assert r.status_code == 201 and r.json()["stream_seconds"] == v and r.json()["stream"]["seconds_limit"] == v
    assert r.json()["stream_max_jobs"] is None and r.json()["stream"]["jobs_limit"] is None


@pytest.mark.parametrize("value", [0, -1, 1001])
def test_stream_max_jobs_out_of_range_is_400(client, value):
    r = create(client, stream_max_jobs=value)
    assert r.status_code == 400 and r.json()["error"] == "invalid_stream_max_jobs" and f"got {value}" in r.json()["detail"]


def test_ibm_session_without_stream_max_jobs_is_400_and_submits_nothing(tmp_path):
    fake = FakeIbm()
    with TestClient(ibm_app(tmp_path, fake)) as c:
        for mode in ("live", "buffered"):  # the mode can be PATCHed to live later, so every ibm session needs it
            r = create(c, provider="ibm", mode=mode)
            assert r.status_code == 400 and r.json()["error"] == "invalid_stream_max_jobs"
            assert "stream_max_jobs is required for ibm" in r.json()["detail"]
        assert create(c, provider="ibm", mode="live", stream_max_jobs=3).status_code == 201
        assert c.get("/api/jobs").json() == []
    assert fake.jobs == 0


# --- limits, mode switches and refills ---------------------------------------------------------------------
import threading

import numpy as np

from quantum_lite.errors import AppError
from quantum_lite.pools import PoolBlock

# auto_refill is sim-only (ibm refuses it at create), so the stream-caused refill gate is exercised on sim.
SIM_PARAMS = {"provider": "sim", "mode": "buffered", "num_qubits": 3, "aggregation": "none", "shots_per_job": 10,
              "interval_ms": 50, "auto_refill": True, "refill_below": 80, "refill_shots": 100,
              "stream_seconds": 1, "stream_max_jobs": 1}


def seed(mgr, s, shots=100):
    mgr.pools.append(s.provider, s.implementation, s.n_qubits,
                     PoolBlock("seed", s.implementation, tuple(s.physical_qubits), "2026-01-01T00:00:00+00:00",
                               np.zeros((shots, s.n_qubits), dtype=np.uint8)))


def watch_submits(mgr):
    """Monotonic time of every job submit the runner accepts."""
    times = []
    real = mgr.runner.submit

    def submit(*a, **k):
        times.append(time.monotonic())
        return real(*a, **k)

    mgr.runner.submit = submit
    return times


def test_buffered_stream_refills_count_against_stream_max_jobs(tmp_path):
    async def main():
        mgr = create_app(make_settings(tmp_path)).state.mgr
        s = await mgr.create({**SIM_PARAMS, "stream_seconds": 5})
        seed(mgr, s)
        await mgr.start_stream(s)
        while s.streaming:
            await asyncio.sleep(0.02)
        st = s.stream_dict()
        assert st["stopped_reason"].startswith("limit_jobs:"), st
        assert st["error"] is None
        assert len(mgr.runner.records) == 1 and st["jobs_used"] == 1  # the second refill was refused, not submitted
        assert s.auto_refill is True  # a limit is not a refill fault
        await asyncio.sleep(0.3)
        assert len(mgr.runner.records) == 1
        await mgr.shutdown()

    asyncio.run(main())


def test_a_pool_take_running_at_the_deadline_completes_is_recorded_and_its_refill_is_refused(tmp_path):
    gate = threading.Event()  # holds the third take past the deadline

    async def main():
        mgr = create_app(make_settings(tmp_path)).state.mgr
        s = await mgr.create({**SIM_PARAMS, "stream_max_jobs": 10})
        seed(mgr, s)
        submits = watch_submits(mgr)
        real_take, calls = mgr.pools.take, []

        def take(key, shots):
            calls.append(key)
            if len(calls) == 3:
                assert gate.wait(10), "the test never released the take"
            return real_take(key, shots)

        mgr.pools.take = take
        try:
            await mgr.start_stream(s)
            deadline = time.monotonic() + s.stream_dict()["seconds_left"]
            while time.monotonic() < deadline + 0.2:
                await asyncio.sleep(0.02)
            assert s.streaming and s.stream_dict()["stopped_reason"] is None  # the take is still held: nothing was given up
            assert s.stats.snapshot()["shots"] == 20
        finally:
            released = time.monotonic()
            gate.set()
        while s.streaming and time.monotonic() - released < 3:
            await asyncio.sleep(0.02)
        assert not s.streaming and time.monotonic() - released < 1.0
        st = s.stream_dict()
        assert st["stopped_reason"].startswith("limit_seconds:") and st["error"] is None
        assert s.stats.snapshot()["shots"] == 30  # the held take's shots were recorded (n grew by shots_per_job)
        assert all(t < deadline for t in submits) and not mgr.runner.records and st["jobs_used"] == 0  # refill refused
        assert s.auto_refill is True
        await mgr.shutdown()

    asyncio.run(main())


def test_start_stream_judges_the_mode_current_at_task_creation(tmp_path):
    """PATCH live while start_stream awaits the pool check: the stream starts live, and its first cycle
    waits for the stopped run's job instead of starting a second stream job beside it."""
    async def main():
        mgr = create_app(make_settings(tmp_path, sim_latency_seconds=1.5)).state.mgr
        s = await mgr.create({"provider": "sim", "mode": "live", "num_qubits": 3, "aggregation": "none",
                              "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": None})
        await mgr.start_stream(s)
        while not mgr.runner.has_inflight(s.id, "stream"):
            await asyncio.sleep(0.01)
        await mgr.stop_stream(s)  # the job cannot be recalled: it stays in flight
        assert mgr.runner.has_inflight(s.id, "stream") and not s.streaming
        mgr.switch_mode(s, "buffered")

        entered, release = threading.Event(), threading.Event()

        def held_available(key):  # runs in the worker thread of start_stream's pool check
            entered.set()
            assert release.wait(10)
            return 1000

        mgr.pools.available = held_available
        start = asyncio.ensure_future(mgr.start_stream(s))
        assert await asyncio.to_thread(entered.wait, 5)
        mgr.switch_mode(s, "live")  # allowed: no stream is running yet
        release.set()
        await start
        assert s.streaming and s.segment.mode == "live"

        def stream_jobs():
            return [r for r in mgr.runner.records.values() if r.kind == "stream"]

        max_in_flight, bound = 0, time.monotonic() + 10
        while len(stream_jobs()) < 2 or stream_jobs()[1].finished_at is None:  # the new run's first job done
            max_in_flight = max(max_in_flight, sum(r.finished_at is None for r in stream_jobs()))
            assert s.streaming and time.monotonic() < bound, s.stream_dict()
            await asyncio.sleep(0.01)
        assert max_in_flight == 1
        first, second = stream_jobs()[:2]
        assert first.discarded and not second.discarded
        assert datetime.fromisoformat(second.submitted_at) >= datetime.fromisoformat(first.finished_at)
        await mgr.stop_stream(s)
        await mgr.shutdown()

    asyncio.run(main())


def test_two_concurrent_starts_make_one_stream(tmp_path):
    async def main():
        mgr = create_app(make_settings(tmp_path)).state.mgr
        s = await mgr.create({"provider": "sim", "mode": "buffered", "num_qubits": 3, "aggregation": "none",
                              "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": None})
        seed_pool = PoolBlock("seed", s.implementation, tuple(s.physical_qubits), "2026-01-01T00:00:00+00:00",
                              np.zeros((500, 3), dtype=np.uint8))
        mgr.pools.append(s.provider, s.implementation, 3, seed_pool)
        results = await asyncio.gather(mgr.start_stream(s), mgr.start_stream(s), return_exceptions=True)
        assert sorted(type(r).__name__ for r in results) == ["AppError", "NoneType"]
        err = next(r for r in results if isinstance(r, AppError))
        assert err.code == "stream_running"
        await mgr.stop_stream(s)
        await mgr.shutdown()

    asyncio.run(main())


# --- deadline races ---------------------------------------------------------------------
def test_ibm_session_with_auto_refill_is_400_and_sim_still_accepts_it(tmp_path):
    fake = FakeIbm()
    with TestClient(ibm_app(tmp_path, fake)) as c:
        body = {"provider": "ibm", "mode": "buffered", "num_qubits": 3, "aggregation": "none", "shots_per_job": 10,
                "interval_ms": 20, "auto_refill": True, "refill_below": 5, "stream_seconds": 30, "stream_max_jobs": 2}
        r = c.post("/api/sessions", json=body)
        assert r.status_code == 400 and r.json()["error"] == "auto_refill_ibm", r.text
        assert "billed" in r.text and "prefetch" in r.text
        assert fake.jobs == 0
        assert c.post("/api/sessions", json={**body, "auto_refill": False}).status_code == 201
        assert c.post("/api/sessions", json={**body, "provider": "sim", "stream_max_jobs": None}).status_code == 201


def test_a_live_cycle_past_the_deadline_is_refused_at_the_permit_and_submits_nothing(tmp_path):
    """The loop's pre-check passed, then the loop stalled past the deadline: the permit itself refuses."""
    fake = FakeIbm()

    async def main():
        mgr = ibm_app(tmp_path, fake).state.mgr
        s = await mgr.create({"provider": "ibm", "mode": "live", "num_qubits": 3, "aggregation": "none",
                              "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 1, "stream_max_jobs": 5})
        now = [100.0]
        s.run = run = StreamRun(1, 5, lambda: now[0])
        now[0] += 1.2  # the stall
        sub = mgr.bus.subscribe(s.id)
        c = await mgr.acq.cycle(s, run)
        assert c == Cycle(None, Stop.REFUSED)
        assert run.refusal[0] == "limit_seconds" and run.jobs_used == 0
        assert fake.jobs == 0 and not mgr.runner.records
        assert sub.queue.empty()  # no `stream` event, no job event
        await mgr.shutdown()

    asyncio.run(main())


def test_a_live_cycle_submits_counts_and_announces_its_job_before_it_first_suspends(tmp_path):
    """Nothing may run between the loop's deadline check and the permit: an await or a task before the
    submit would let a stalled loop pass the deadline (the check sits at the submit itself). With no
    stream job of the session in flight there is nothing to wait for, so the cycle does not suspend."""
    async def main():
        mgr = create_app(make_settings(tmp_path)).state.mgr
        s = await mgr.create({"provider": "sim", "mode": "live", "num_qubits": 3, "aggregation": "none",
                              "shots_per_job": 10, "interval_ms": 20, "auto_refill": False, "refill_below": 0,
                              "refill_shots": None, "stream_seconds": 60, "stream_max_jobs": None})
        s.run = run = StreamRun(60, None, mgr.clock)  # as start_stream sets it: the `stream` event reads it
        sub = mgr.bus.subscribe(s.id)
        coro = mgr.acq.cycle(s, run)
        coro.send(None)  # runs the cycle up to its first suspension; nothing else on the loop has run
        [rec] = mgr.runner.records.values()
        assert rec.kind == "stream" and rec.session_id == s.id and run.jobs_used == 1
        events = [sub.queue.get_nowait() for _ in range(sub.queue.qsize())]
        assert [(e, d.get("state")) for e, d in events] == [("job", "submitting"), ("stream", None)]
        assert events[1][1]["stream"]["jobs_used"] == 1
        with pytest.raises(asyncio.CancelledError):
            coro.throw(asyncio.CancelledError())  # the stream is stopped there: the job runs on, is pooled
        await until(lambda: rec.pooled_to is not None, "the discarded job to be pooled")
        assert rec.discarded and rec.state == "done" and rec.pooled_to == s.pool_key
        assert s.stats.snapshot()["shots"] == 0
        await mgr.shutdown()

    asyncio.run(main())
