"""Acquisition through its interface: one, cycle, Cycle.pace, admit_stream; depth 1 at the permit.

Real runner, feeder and bus, wired as create_app wires them, over a tmp pool store. Bits come
from a provider (sim, gated, scripted, ibm over FakeIbm) or from a seeded pool, never from a
stand-in for acquisition itself.
"""
import asyncio
import threading
from datetime import datetime

import numpy as np
import pytest

from conftest import make_settings, until
from fakes import FakeIbm, ScriptedSim
from quantum_lite.acquisition import Cycle, Stop
from quantum_lite.app import create_app
from quantum_lite.errors import AppError
from quantum_lite.pools import PoolBlock, PoolStore
from quantum_lite.providers.ibm import IbmProvider
from quantum_lite.stream_run import StreamRun
from test_event_payloads import SIM_JOB, TS, Events, GatedSim, Like, same

SESSION = {"provider": "sim", "mode": "live", "num_qubits": 3, "aggregation": "none", "shots_per_job": 10,
           "interval_ms": 20, "auto_refill": False, "refill_below": 0, "refill_shots": None,
           "stream_seconds": 60, "stream_max_jobs": None}
DAY1, DAY2 = "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"


def manager(tmp_path, sim=None):
    settings = make_settings(tmp_path)
    app = create_app(settings) if sim is None else create_app(settings, providers={"sim": sim(settings)})
    return app.state.mgr


def seed(mgr, job_id, shots, at=DAY1, n=3, bits=None):
    bits = np.zeros((shots, n), np.uint8) if bits is None else np.asarray(bits, np.uint8)
    mgr.pools.append("sim", "aer_ideal", n, PoolBlock(job_id, "aer_ideal", tuple(range(n)), at, bits))


def spent_run(clock, max_jobs=1):
    """A run whose job limit is used up: the next submit through it is refused."""
    run = StreamRun(60, max_jobs, clock)
    for _ in range(max_jobs):
        run.submit(lambda: None)
    return run


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# --- one ------------------------------------------------------------------------------------------
def test_one_live_provenance_literal(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        s = await mgr.create(SESSION)
        batch = await mgr.acq.one(s, "predict")
        p = batch.provenance
        same(p, {"job_id": Like(SIM_JOB), "job_ids": [p["job_id"]], "source": "live", "provider": "sim",
                 "backend": "aer_ideal", "measured_at": Like(TS), "physical_qubits": [0, 1, 2],
                 "pieces": [{"job_id": p["job_id"], "backend": "aer_ideal", "measured_at": p["measured_at"],
                             "physical_qubits": [0, 1, 2], "shots": 1}],
                 "segment": 0, "shots": 1})
        assert batch.bits.shape == (1, 3) and batch.segment is s.segments[0]
        [rec] = mgr.runner.records.values()
        assert rec.kind == "predict" and rec.shots == 1 and rec.job_id == p["job_id"]
        await mgr.shutdown()

    asyncio.run(main())


def test_one_buffered_provenance_literal_and_aggregation(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 1, DAY1, bits=[[1, 1, 0]])
        seed(mgr, "j2", 1, DAY2, bits=[[0, 1, 1]])
        s = await mgr.create({**SESSION, "mode": "buffered", "aggregation": "majority"})
        first, second = await mgr.acq.one(s, "press"), await mgr.acq.one(s, "perceive")
        same(second.provenance, {"job_id": "j2", "job_ids": ["j2"], "source": "buffered", "provider": "sim",
                                 "backend": "aer_ideal", "measured_at": DAY2, "physical_qubits": [0, 1, 2],
                                 "pieces": [{"job_id": "j2", "backend": "aer_ideal", "measured_at": DAY2,
                                             "physical_qubits": [0, 1, 2], "shots": 1}],
                                 "segment": 0, "shots": 1})
        assert first.provenance["job_id"] == "j1"
        assert first.bits.tolist() == [[1, 1, 0]] and first.aggregated.tolist() == [1]
        assert second.bits.tolist() == [[0, 1, 1]] and second.aggregated.tolist() == [1]
        assert mgr.runner.records == {}
        await mgr.shutdown()

    asyncio.run(main())


def test_one_buffered_pool_empty_starts_the_refill_before_the_409(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        s = await mgr.create({**SESSION, "mode": "buffered", "auto_refill": True, "refill_below": 50,
                              "refill_shots": 50})
        ev = Events(mgr.bus.subscribe(s.id))
        with pytest.raises(AppError) as err:
            await mgr.acq.one(s, "press")
        assert err.value.status == 409 and err.value.code == "pool_empty"
        assert [(e, d.get("state"), d.get("kind")) for e, d in ev.pump()] == [("job", "submitting", "prefetch")]
        await mgr.feeder.drain()
        assert mgr.pools.available(s.pool_key) == 50
        await mgr.shutdown()

    asyncio.run(main())


def test_one_ibm_job_error_is_502_and_the_simulator_is_never_called(tmp_path):
    async def main():
        settings = make_settings(tmp_path, ibm_token="SECRET-TOKEN-0123456789")
        fake = FakeIbm(statuses=("ERROR",), error="device offline")
        sim = ScriptedSim(settings, [[1]])
        ibm = IbmProvider(settings, service_factory=fake.service_factory, sampler_cls=fake.sampler_cls,
                          poll_seconds=0.01)
        mgr = create_app(settings, providers={"sim": sim, "ibm": ibm}).state.mgr
        s = await mgr.create({**SESSION, "provider": "ibm", "stream_max_jobs": 5})
        with pytest.raises(AppError) as err:
            await mgr.acq.one(s, "press")
        assert (err.value.status, err.value.code) == (502, "provider_error")
        assert err.value.detail == "ProviderError: IBM job fakejob-1 ended error: device offline"
        assert fake.jobs == 1 and sim.jobs == 0 and mgr.pools.list() == []
        await mgr.shutdown()

    asyncio.run(main())


def test_one_keeps_the_segment_it_started_in_across_a_switch(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        s = await mgr.create(SESSION)
        try:
            press = asyncio.ensure_future(mgr.acq.one(s, "press"))
            await until(lambda: mgr.runner.has_inflight(s.id, "press"), "the press job")
            mgr.switch_mode(s, "buffered")
        finally:
            sim.gate.set()
        batch = await press
        assert batch.segment is s.segments[0] and s.segment is s.segments[1]
        assert (batch.provenance["source"], batch.provenance["segment"]) == ("live", 0)
        await mgr.shutdown()

    asyncio.run(main())


def test_one_buffered_take_keeps_its_segment_across_a_switch(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 5)
        s = await mgr.create({**SESSION, "mode": "buffered"})
        entered, release = threading.Event(), threading.Event()
        real_take = mgr.pools.take

        def held_take(key, shots):
            entered.set()
            assert release.wait(10)
            return real_take(key, shots)

        mgr.pools.take = held_take
        try:
            press = asyncio.ensure_future(mgr.acq.one(s, "press"))
            assert await asyncio.to_thread(entered.wait, 5)
            mgr.switch_mode(s, "live")
        finally:
            release.set()
        batch = await press
        assert batch.segment is s.segments[0]
        assert (batch.provenance["source"], batch.provenance["job_id"]) == ("buffered", "j1")
        assert mgr.runner.records == {}  # no live job was submitted for it
        await mgr.shutdown()

    asyncio.run(main())


# --- cycle ----------------------------------------------------------------------------------------
def test_live_cycle_at_the_deadline_is_abandoned_and_its_job_pooled_uncounted(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        s = await mgr.create(SESSION)
        clock = Clock()
        s.run = run = StreamRun(1, None, clock)
        clock.t = run.deadline - 0.05
        try:
            c = await mgr.acq.cycle(s, run)
        finally:
            sim.gate.set()
        assert c == Cycle(None, Stop.ABANDONED) and run.refusal is None and run.jobs_used == 1
        [rec] = mgr.runner.records.values()
        await until(lambda: rec.pooled_to is not None, "the abandoned job to be pooled")
        assert rec.discarded and rec.kind == "stream" and rec.pooled_to == s.pool_key
        assert mgr.pools.available(s.pool_key) == 10 and s.stats.snapshot()["shots"] == 0
        await mgr.shutdown()

    asyncio.run(main())


def test_live_cycle_keeps_the_segment_it_started_in_across_a_switch(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        s = await mgr.create(SESSION)
        s.run = run = StreamRun(60, None, mgr.clock)
        try:
            cycle = asyncio.ensure_future(mgr.acq.cycle(s, run))
            await until(lambda: mgr.runner.has_inflight(s.id, "stream"), "the stream job")
            mgr.switch_mode(s, "buffered")
        finally:
            sim.gate.set()
        c = await cycle
        assert c.stop is None and c.batch.segment is s.segments[0]
        assert (c.batch.provenance["source"], c.batch.provenance["segment"]) == ("live", 0)
        await mgr.shutdown()

    asyncio.run(main())


def test_live_cycle_delivers_the_scripted_bits(tmp_path):
    async def main():
        bits = np.array([[1, 0, 0]] * 4 + [[0, 1, 1]] * 6, dtype=np.uint8)
        mgr = manager(tmp_path, sim=lambda st: ScriptedSim(st, [bits]))
        s = await mgr.create(SESSION)
        s.run = run = StreamRun(60, None, mgr.clock)
        c = await mgr.acq.cycle(s, run)
        assert c.stop is None and c.batch.bits.tolist() == bits.tolist()
        assert c.batch.aggregated.tolist() == [1] * 4 + [0] * 6 and c.batch.segment is s.segments[0]
        await mgr.shutdown()

    asyncio.run(main())


def test_buffered_take_held_past_the_deadline_still_returns_its_batch(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 30)
        s = await mgr.create({**SESSION, "mode": "buffered"})
        clock = Clock()
        s.run = run = StreamRun(1, None, clock)
        real_take = mgr.pools.take

        def late_take(key, shots):
            clock.t = run.deadline + 5  # the deadline passes while the take runs
            return real_take(key, shots)

        mgr.pools.take = late_take
        c = await mgr.acq.cycle(s, run)
        assert c.stop is None and c.batch.provenance["shots"] == 10 and c.batch.provenance["source"] == "buffered"
        assert mgr.pools.get(s.pool_key).cursor == 10
        await mgr.shutdown()

    asyncio.run(main())


def test_buffered_cycle_whose_refill_is_refused_delivers_its_batch_with_the_stop(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 85)
        s = await mgr.create({**SESSION, "mode": "buffered", "auto_refill": True, "refill_below": 80,
                              "refill_shots": 100})
        s.run = run = spent_run(mgr.clock)
        c = await mgr.acq.cycle(s, run)
        assert c.stop is Stop.REFUSED and c.batch is not None and c.batch.provenance["shots"] == 10
        assert run.refusal[0] == "limit_jobs" and run.jobs_used == 1 and mgr.runner.records == {}
        assert s.auto_refill is True  # a limit is not a refill fault
        await mgr.shutdown()

    asyncio.run(main())


def test_buffered_cycle_on_an_empty_pool_is_a_stop_when_its_refill_was_refused_else_a_409(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        refilled = await mgr.create({**SESSION, "mode": "buffered", "auto_refill": True, "refill_below": 50,
                                     "refill_shots": 50})
        refilled.run = run = spent_run(mgr.clock)
        assert await mgr.acq.cycle(refilled, run) == Cycle(None, Stop.REFUSED)
        assert run.refusal[0] == "limit_jobs" and mgr.runner.records == {}
        plain = await mgr.create({**SESSION, "mode": "buffered"})
        plain.run = run = StreamRun(60, None, mgr.clock)
        with pytest.raises(AppError) as err:
            await mgr.acq.cycle(plain, run)
        assert err.value.code == "pool_empty" and run.refusal is None and run.jobs_used == 0
        await mgr.shutdown()

    asyncio.run(main())


def test_buffered_cycle_on_an_empty_pool_submits_no_stream_job(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        s = await mgr.create({**SESSION, "mode": "buffered", "auto_refill": True, "refill_below": 50,
                              "refill_shots": 50})
        s.run = run = StreamRun(60, 5, mgr.clock)
        with pytest.raises(AppError) as err:
            await mgr.acq.cycle(s, run)
        assert err.value.code == "pool_empty"
        assert [r.kind for r in mgr.runner.records.values()] == ["prefetch"] and run.jobs_used == 1
        assert not mgr.runner.has_inflight(s.id, "stream")
        await mgr.shutdown()

    asyncio.run(main())


# --- pace -----------------------------------------------------------------------------------------
def test_live_pace_returns_without_suspending(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        s = await mgr.create(SESSION)
        s.run = run = StreamRun(60, None, mgr.clock)
        c = await mgr.acq.cycle(s, run)
        coro = c.pace(mgr.clock())
        with pytest.raises(StopIteration):
            coro.send(None)
        await mgr.shutdown()

    asyncio.run(main())


def test_buffered_pace_waits_out_the_interval_clamped_to_the_deadline(tmp_path, monkeypatch):
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 100)
        s = await mgr.create({**SESSION, "mode": "buffered", "interval_ms": 300})
        clock = Clock()
        s.run = run = StreamRun(60, None, clock)
        t0 = clock()
        c = await mgr.acq.cycle(s, run)
        slept, real_sleep = [], asyncio.sleep

        async def sleep(seconds):  # the arithmetic is under test, not the scheduler: record, do not wait
            slept.append(seconds)
            await real_sleep(0)

        monkeypatch.setattr("quantum_lite.acquisition.asyncio.sleep", sleep)
        await c.pace(t0)  # the run's clock stood still: the whole 0.3 s interval
        clock.t = t0 + 0.1  # 0.1 s of the interval already spent in the cycle
        await c.pace(t0)
        clock.t = run.deadline - 0.05  # the deadline comes before the interval ends
        await c.pace(clock.t)
        clock.t = t0 + 0.5  # the cycle took longer than the interval
        await c.pace(t0)
        monkeypatch.undo()
        assert slept == [pytest.approx(0.3), pytest.approx(0.2), pytest.approx(0.05), 0.0]
        await mgr.shutdown()

    asyncio.run(main())


# --- admit_stream --------------------------------------------------------------------------------
def test_admit_stream_live_returns_without_suspending(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        s = await mgr.create(SESSION)
        coro = mgr.acq.admit_stream(s)
        with pytest.raises(StopIteration):
            coro.send(None)
        await mgr.shutdown()

    asyncio.run(main())


def test_admit_stream_on_a_short_pool_is_409_with_the_text_and_starts_a_refill(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 5)
        s = await mgr.create({**SESSION, "mode": "buffered", "auto_refill": True, "refill_below": 1,
                              "refill_shots": 50})
        with pytest.raises(AppError) as err:
            await mgr.acq.admit_stream(s)
        assert (err.value.status, err.value.code) == (409, "pool_empty")
        assert err.value.detail == "pool sim__aer_ideal__3 has 5 unconsumed shots, stream needs 10 per cycle"
        [rec] = mgr.runner.records.values()
        assert rec.kind == "prefetch" and rec.shots == 50 and rec.session_id == s.id
        await mgr.feeder.drain()
        await mgr.acq.admit_stream(s)  # the refill landed: admitted
        await mgr.shutdown()

    asyncio.run(main())


async def _stream_job_held(mgr, sim, s):
    """Start a live stream, wait for its job (held by the gate), stop the stream: the job stays in flight."""
    await mgr.start_stream(s)
    await until(lambda: mgr.runner.has_inflight(s.id, "stream"), "the stream job")
    await mgr.stop_stream(s)


def test_admit_stream_live_with_a_stream_job_in_flight_admits_without_suspending(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        s = await mgr.create(SESSION)
        try:
            await _stream_job_held(mgr, sim, s)
            coro = mgr.acq.admit_stream(s)  # depth 1 is kept by the cycle, at the permit
            with pytest.raises(StopIteration):
                coro.send(None)
        finally:
            sim.gate.set()
        await mgr.shutdown()

    asyncio.run(main())


# --- depth 1 at the permit ------------------------------------------------------------------------
def test_live_cycle_waits_for_the_previous_stream_job_then_submits(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        s = await mgr.create(SESSION)
        try:
            await _stream_job_held(mgr, sim, s)
            [held] = mgr.runner.records.values()
            s.run = run = StreamRun(60, None, mgr.clock)
            cycle = asyncio.ensure_future(mgr.acq.cycle(s, run))
            await asyncio.sleep(0.2)
            assert not cycle.done() and len(mgr.runner.records) == 1 and run.jobs_used == 0
        finally:
            sim.gate.set()
        c = await cycle
        assert c.stop is None and c.batch.provenance["shots"] == 10 and run.jobs_used == 1
        first, second = mgr.runner.records.values()
        assert first is held and first.discarded and not second.discarded
        assert c.batch.provenance["job_id"] == second.job_id
        assert datetime.fromisoformat(second.submitted_at) >= datetime.fromisoformat(first.finished_at)
        await mgr.shutdown()

    asyncio.run(main())


def test_live_cycle_still_waiting_at_the_deadline_expires_and_submits_nothing(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        s = await mgr.create(SESSION)
        try:
            await _stream_job_held(mgr, sim, s)
            clock = Clock()
            s.run = run = StreamRun(1, None, clock)
            clock.t = run.deadline - 0.05
            assert await mgr.acq.cycle(s, run) == Cycle(None, Stop.EXPIRED)
            assert len(mgr.runner.records) == 1 and run.jobs_used == 0 and run.refusal is None
            s2 = await mgr.create({**SESSION, "stream_seconds": 1})
            await _stream_job_held(mgr, sim, s2)
            ev = Events(mgr.bus.subscribe(s2.id))
            await mgr.start_stream(s2)
            await until(lambda: not s2.streaming, "the run to end at its deadline")
            [(_, stopped)] = ev.of("stream_stopped")
            assert (stopped["code"], stopped["detail"]) == ("limit_seconds", "the 1 s stream duration is over")
            assert s2.stream_dict()["stopped_reason"] == "limit_seconds: the 1 s stream duration is over"
            assert len(mgr.runner.records) == 2 and s2.stream_dict()["jobs_used"] == 0
        finally:
            sim.gate.set()
        await mgr.shutdown()

    asyncio.run(main())


def test_a_stream_job_in_flight_at_the_permit_is_a_409_never_a_second_job(tmp_path):
    """The guard inside the permit: if the wait ever let a cycle through with a job in flight."""
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        s = await mgr.create(SESSION)
        try:
            await _stream_job_held(mgr, sim, s)

            async def settled(session_id, kind, timeout):  # a broken wait
                return True

            mgr.runner.settled = settled
            s.run = run = StreamRun(60, None, mgr.clock)
            with pytest.raises(AppError) as err:
                await mgr.acq.cycle(s, run)
            assert (err.value.status, err.value.code) == (409, "stream_job_in_flight")
            assert len(mgr.runner.records) == 1 and run.jobs_used == 0
        finally:
            sim.gate.set()
        await mgr.shutdown()

    asyncio.run(main())


def test_a_switch_to_buffered_while_a_live_cycle_waits_is_obeyed_and_submits_no_live_job(tmp_path):
    """The depth-1 wait comes before the capture: the cycle takes from the pool once the held job
    finishes, and no stream job is submitted after the user left live mode."""
    async def main():
        settings = make_settings(tmp_path)
        sim = GatedSim(settings)
        mgr = create_app(settings, providers={"sim": sim}).state.mgr
        seed(mgr, "j1", 100)
        s = await mgr.create(SESSION)
        try:
            await _stream_job_held(mgr, sim, s)
            [held] = mgr.runner.records.values()
            s.run = run = StreamRun(60, None, mgr.clock)
            cycle = asyncio.ensure_future(mgr.acq.cycle(s, run))
            await asyncio.sleep(0.1)
            assert not cycle.done()  # waiting for the held job
            mgr.switch_mode(s, "buffered")
        finally:
            sim.gate.set()
        c = await cycle
        assert c.stop is None and c.batch.segment is s.segments[1]
        assert (c.batch.provenance["source"], c.batch.provenance["segment"], c.batch.provenance["job_id"]) == \
            ("buffered", 1, "j1")
        assert list(mgr.runner.records.values()) == [held] and run.jobs_used == 0
        await until(lambda: held.pooled_to is not None, "the held job to be pooled")
        await mgr.shutdown()

    asyncio.run(main())


def test_a_stop_landing_just_after_the_live_job_finished_pools_its_bits(tmp_path):
    """The job is done but the cycle is cancelled before it takes the result: the bits are pooled."""
    async def main():
        mgr = manager(tmp_path)
        s = await mgr.create(SESSION)
        s.run = run = StreamRun(60, None, mgr.clock)
        real_start = mgr.runner.start

        def start(rec, provider):  # stop_stream lands right after the job finished
            job = real_start(rec, provider)
            job.add_done_callback(lambda _: cycle.cancel())
            return job

        mgr.runner.start = start
        cycle = asyncio.ensure_future(mgr.acq.cycle(s, run))
        with pytest.raises(asyncio.CancelledError):
            await cycle
        [rec] = mgr.runner.records.values()
        await until(lambda: rec.pooled_to is not None, "the finished job to be pooled")
        assert rec.state == "done" and rec.discarded and rec.pooled_to == s.pool_key
        assert mgr.pools.available(s.pool_key) == 10 and s.stats.snapshot()["shots"] == 0
        await mgr.shutdown()

    asyncio.run(main())



# --- a stop never drops shots already taken from the pool ------------------------------------
def _held(fn):
    """Wrap a blocking pool call so it waits for `release`; `entered` is set once it is waiting."""
    entered, release = threading.Event(), threading.Event()

    def held(*args):
        entered.set()
        assert release.wait(10), "held pool call not released"
        return fn(*args)

    return held, entered, release


def _stopped_while_held(tmp_path, stop, *, hold="take", get_fails=False):
    """Hold a buffered stream's first cycle in `hold`, stop the stream with `stop`, then release it.

    `get_fails`: the pool re-read after the take raises, as a broken store would.
    """
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 100)
        s = await mgr.create({**SESSION, "mode": "buffered", "auto_refill": True, "refill_below": 5,
                              "refill_shots": 50})
        ev = Events(mgr.bus.subscribe(s.id))
        if get_fails:
            def broken_get(key):
                raise PermissionError("pool directory unreadable")

            mgr.pools.get = broken_get
        if hold == "take":
            mgr.pools.take, entered, release = _held(mgr.pools.take)
        else:  # the refill check that follows a successful take
            real_take, real_available = mgr.pools.take, mgr.pools.available
            held_available, entered, release = _held(real_available)
            taken = threading.Event()

            def take(*args):
                blocks = real_take(*args)
                taken.set()
                return blocks

            mgr.pools.take = take
            mgr.pools.available = lambda key: held_available(key) if taken.is_set() else real_available(key)
        await mgr.start_stream(s)
        assert await asyncio.to_thread(entered.wait, 5), "the cycle never reached the held call"
        stopping = asyncio.ensure_future(stop(mgr, s))
        await asyncio.sleep(0.05)
        assert not stopping.done()  # the stop waits for the shots already being taken
        release.set()
        await asyncio.wait_for(stopping, 5)
        assert PoolStore(mgr.pools.dir.parent).get(s.pool_key).cursor == 10
        assert s.mode_stats["buffered"].snapshot()["shots"] == 10
        batches = ev.of("batch")
        assert len(batches) == 1 and batches[0][1]["provenance"]["job_id"] == "j1"
        assert batches[0][1]["shots"] == 10
        errors = ev.of("error")
        if get_fails:
            assert [(d["scope"], d["error"]) for _, d in errors] == [("pool", "pool_read_failed")]
            assert "PermissionError: pool directory unreadable" in errors[0][1]["detail"]
        else:
            assert errors == []
        assert not s.streaming
        await mgr.shutdown()

    asyncio.run(main())


def test_a_stop_during_a_buffered_take_records_the_shots_taken(tmp_path):
    _stopped_while_held(tmp_path, lambda mgr, s: mgr.stop_stream(s))


def test_a_stop_after_the_take_during_its_refill_check_records_the_shots_taken(tmp_path):
    _stopped_while_held(tmp_path, lambda mgr, s: mgr.stop_stream(s), hold="available")


def test_a_shutdown_during_a_buffered_take_records_the_shots_taken(tmp_path):
    _stopped_while_held(tmp_path, lambda mgr, s: mgr.shutdown())


def test_a_second_cancel_while_the_take_finishes_still_records_the_shots(tmp_path):
    async def stop_twice(mgr, s):
        task = s.stream_task
        stopping = asyncio.ensure_future(mgr.stop_stream(s))
        await asyncio.sleep(0.01)
        task.cancel()  # e.g. the last listener goes, or shutdown, while the stop waits
        await stopping

    _stopped_while_held(tmp_path, stop_twice)


def test_a_stop_during_a_take_whose_pool_re_read_fails_still_records_the_shots(tmp_path):
    _stopped_while_held(tmp_path, lambda mgr, s: mgr.stop_stream(s), get_fails=True)


def test_a_cycle_whose_pool_re_read_fails_delivers_its_batch_and_reports_the_failure(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 30)
        s = await mgr.create({**SESSION, "mode": "buffered"})
        ev = Events(mgr.bus.subscribe(s.id))

        def broken_get(key):
            raise PermissionError("pool directory unreadable")

        mgr.pools.get = broken_get
        c = await mgr.acq.cycle(s, StreamRun(60, None, Clock()))
        assert c.stop is None and c.batch.provenance["shots"] == 10
        assert [(d["scope"], d["error"]) for _, d in ev.of("error")] == [("pool", "pool_read_failed")]
        await mgr.shutdown()

    asyncio.run(main())


def test_a_cycle_whose_refill_check_fails_delivers_its_batch_and_reports_the_failure(tmp_path):
    async def main():
        mgr = manager(tmp_path)
        seed(mgr, "j1", 30)
        s = await mgr.create({**SESSION, "mode": "buffered", "auto_refill": True, "refill_below": 5,
                              "refill_shots": 50})
        ev = Events(mgr.bus.subscribe(s.id))

        def broken_available(key):
            raise PermissionError("pool directory unreadable")

        mgr.pools.available = broken_available
        c = await mgr.acq.cycle(s, StreamRun(60, None, Clock()))
        assert c.stop is None and c.batch.provenance["shots"] == 10
        assert [(d["scope"], d["error"]) for _, d in ev.of("error")] == [("pool", "refill_check_failed")]
        await mgr.shutdown()

    asyncio.run(main())
