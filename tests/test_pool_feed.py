"""PoolFeeder and the sync core, on a real tmp PoolStore, JobRunner, EventBus and IbmBudget.

Failure, slowness and a device mismatch come from small Provider adapters, not from patching
the runner.
"""
import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from conftest import make_settings
from fakes import FakeIbm
from quantum_lite.budget import IbmBudget
from quantum_lite.errors import AppError
from quantum_lite.events import EventBus
from quantum_lite.jobs import JobRunner
from quantum_lite.pool_feed import PoolFeeder, TopUp, bank, check_prefetch, lookup_provider
from quantum_lite.pools import PoolBlock, PoolStore
from quantum_lite.providers.base import Measurement
from quantum_lite.providers.ibm import IbmProvider
from quantum_lite.providers.sim import SimProvider
from quantum_lite.stream_run import StreamRun

SUFFIX = " (auto_refill switched off for this session; no automatic retry)"


class FailingSim(SimProvider):
    def __init__(self, settings):
        super().__init__(settings)
        self.calls = 0

    def run(self, n, shots, on_state, stop):
        self.calls += 1
        raise RuntimeError("device error")


class ElsewhereSim(SimProvider):
    """Resolves to aer_ideal, measures on `other`."""

    def run(self, n, shots, on_state, stop):
        return measurement(n, shots, backend="other", job_id="sim-elsewhere")


class GatedSim(SimProvider):
    """Every job sets `entered` once running, then waits for `gate` before it measures."""

    def __init__(self, settings):
        super().__init__(settings)
        self.entered = threading.Event()
        self.gate = threading.Event()

    def run(self, n, shots, on_state, stop):
        self.entered.set()
        assert self.gate.wait(10), "the test never opened the gate"
        return self.measure(n, shots, on_state, stop)

    def measure(self, n, shots, on_state, stop):
        return super().run(n, shots, on_state, stop)


class GatedElsewhereSim(GatedSim):
    """GatedSim that measures on `other`, like ElsewhereSim."""

    def measure(self, n, shots, on_state, stop):
        return measurement(n, shots, backend="other", job_id="sim-elsewhere")


@dataclass
class Terms:
    """A session's refill terms (RefillTerms)."""

    id: str = "s1"
    provider: str = "sim"
    implementation: str = "aer_ideal"
    n_qubits: int = 5
    pool_key: str = "sim__aer_ideal__5"
    refill_below: int = 50
    refill_shots: int = 50
    auto_refill: bool = True


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def measurement(n, shots, backend="aer_ideal", job_id="sim-x", ones=False):
    bits = (np.ones if ones else np.zeros)((shots, n), np.uint8)
    return Measurement(bits, job_id, "sim", backend, tuple(range(n)), datetime.now(timezone.utc).isoformat())


def feeder_for(tmp_path, sim_cls=SimProvider, **settings_kw):
    settings = make_settings(tmp_path, **settings_kw)
    bus, budget = EventBus(), IbmBudget(settings)
    runner = JobRunner(bus, budget, settings.sim_latency_seconds)
    sim = sim_cls(settings)
    return PoolFeeder({"sim": sim}, PoolStore(settings.data_dir), runner, bus, budget), sim


def seed(store, shots, n=5):
    store.append("sim", "aer_ideal", n, PoolBlock("seed", "aer_ideal", tuple(range(n)), "2026-01-01T00:00:00+00:00",
                                                  np.zeros((shots, n), np.uint8)))


def spy_available(feeder):
    calls = []
    real = feeder.store.available

    def available(key):
        calls.append(key)
        return real(key)

    feeder.store.available = available
    return calls


def drain_events(sub):
    out = []
    while not sub.queue.empty():
        out.append(sub.queue.get_nowait())
    return out


async def until(predicate, what, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


def prefetch_records(feeder):
    return [r for r in feeder.runner.records.values() if r.kind == "prefetch"]


# --- top_up outcomes --------------------------------------------------------------------------
def test_top_up_is_idle_at_refill_below_and_submits_below_it_then_holds_the_guard(tmp_path):
    async def main():
        feeder, _ = feeder_for(tmp_path)
        seed(feeder.store, 50)
        terms = Terms()
        assert await feeder.top_up(terms) is TopUp.IDLE and feeder.runner.records == {}
        feeder.store.take(terms.pool_key, 1)  # 49 < refill_below
        assert await feeder.top_up(terms) is TopUp.SUBMITTED
        assert await feeder.top_up(terms, force=True) is TopUp.BUSY  # its landing has not run yet
        await feeder.drain()
        assert feeder.store.available(terms.pool_key) == 99
        assert len(prefetch_records(feeder)) == 1 and prefetch_records(feeder)[0].session_id == "s1"
        assert await feeder.top_up(terms, force=True) is TopUp.SUBMITTED  # landed: the guard is free again
        await feeder.drain()

    asyncio.run(main())


def test_top_up_with_auto_refill_off_is_idle_and_reads_nothing(tmp_path):
    async def main():
        feeder, _ = feeder_for(tmp_path)
        reads = spy_available(feeder)
        terms = Terms(auto_refill=False)
        assert await feeder.top_up(terms) is TopUp.IDLE
        assert await feeder.top_up(terms, force=True) is TopUp.IDLE
        assert reads == [] and feeder.runner.records == {}

    asyncio.run(main())


def test_force_skips_the_pool_read(tmp_path):
    async def main():
        feeder, _ = feeder_for(tmp_path)
        seed(feeder.store, 500)  # well above refill_below
        reads = spy_available(feeder)
        assert await feeder.top_up(Terms(), force=True) is TopUp.SUBMITTED
        assert reads == []
        await feeder.drain()
        assert await feeder.top_up(Terms()) is TopUp.IDLE and reads == ["sim__aer_ideal__5"]

    asyncio.run(main())


def stopped_by_deadline():
    clock = Clock()
    run = StreamRun(5, None, clock)
    clock.now = 105.0
    return run, ("limit_seconds", "the 5 s stream duration is over"), 0


def stopped_by_jobs():
    run = StreamRun(60, 1, Clock())
    run.submit(lambda: None)
    return run, ("limit_jobs", "1 jobs caused by this stream, the limit is 1"), 1


@pytest.mark.parametrize("make_run", [stopped_by_deadline, stopped_by_jobs])
def test_a_refill_a_stream_limit_refuses_records_nothing_and_leaves_auto_refill_on(tmp_path, make_run):
    async def main():
        feeder, _ = feeder_for(tmp_path)
        run, refusal, used = make_run()
        terms = Terms()
        assert await feeder.top_up(terms, force=True, run=run) is TopUp.REFUSED_BY_RUN
        assert feeder.runner.records == {} and run.jobs_used == used
        assert terms.auto_refill is True and run.refusal == refusal
        assert await feeder.top_up(terms, force=True) is TopUp.SUBMITTED  # no guard was taken
        await feeder.drain()

    asyncio.run(main())


def test_a_refill_a_stream_permits_is_counted_on_its_run(tmp_path):
    async def main():
        feeder, _ = feeder_for(tmp_path)
        run = StreamRun(60, 3, Clock())
        assert await feeder.top_up(Terms(), force=True, run=run) is TopUp.SUBMITTED
        assert run.jobs_used == 1 and len(feeder.runner.records) == 1 and run.refusal is None
        assert await feeder.top_up(Terms(), force=True, run=run) is TopUp.BUSY and run.jobs_used == 1
        await feeder.drain()

    asyncio.run(main())


def test_a_refused_submit_switches_auto_refill_off_and_says_so(tmp_path):
    async def main():
        feeder, _ = feeder_for(tmp_path)

        def refuse(*a, **k):
            raise AppError(429, "job_ceiling", "IBM_MAX_JOBS is 1, 1 already submitted by this process")

        feeder.runner.submit = refuse
        sub = feeder.bus.subscribe("s1")
        terms = Terms()
        assert await feeder.top_up(terms, force=True) is TopUp.SWITCHED_OFF
        assert terms.auto_refill is False and feeder.runner.records == {}
        [(event, data)] = drain_events(sub)
        assert (event, data["scope"], data["error"], data["detail"]) == (
            "error", "refill", "job_ceiling", "IBM_MAX_JOBS is 1, 1 already submitted by this process" + SUFFIX)
        assert await feeder.top_up(terms, force=True) is TopUp.IDLE

    asyncio.run(main())


def test_a_refill_that_fails_while_another_request_reads_the_pool_is_not_resubmitted(tmp_path):
    # Request A awaits the pool read; meanwhile refill B fails, switching auto_refill off and
    # releasing the guard. A must see auto_refill off after its await, not submit a second job.
    async def main():
        feeder, sim = feeder_for(tmp_path, FailingSim)
        seed(feeder.store, 10)
        terms = Terms()
        entered, hold = threading.Event(), threading.Event()
        real_available = feeder.store.available

        def held_available(key):
            entered.set()
            hold.wait(5)
            return real_available(key)

        feeder.store.available = held_available
        a = asyncio.create_task(feeder.top_up(terms))
        await until(entered.is_set, "the pool read")
        assert await feeder.top_up(terms, force=True) is TopUp.SUBMITTED  # B: submits, then its landing fails
        await feeder.drain()
        assert terms.auto_refill is False
        hold.set()
        assert await a is TopUp.IDLE
        assert len(prefetch_records(feeder)) == 1 and sim.calls == 1

    asyncio.run(main())


def test_a_refill_started_while_another_request_reads_the_pool_is_not_doubled(tmp_path):
    # Request A awaits the pool read; meanwhile request B submits a refill that is still in flight
    # when A resumes. A must see the guard after its await.
    async def main():
        feeder, sim = feeder_for(tmp_path, GatedSim)
        terms = Terms()
        entered, hold = threading.Event(), threading.Event()
        real_available = feeder.store.available

        def held_available(key):
            entered.set()
            hold.wait(5)
            return real_available(key)

        feeder.store.available = held_available
        a = asyncio.create_task(feeder.top_up(terms))
        await until(entered.is_set, "the pool read")
        try:
            assert await feeder.top_up(terms, force=True) is TopUp.SUBMITTED  # B
            hold.set()
            assert await a is TopUp.BUSY
            assert len(feeder.runner.records) == 1
        finally:
            hold.set()
            sim.gate.set()
        await feeder.drain()

    asyncio.run(main())


def test_thirty_top_ups_against_a_failing_device_submit_once(tmp_path):
    async def main():
        feeder, sim = feeder_for(tmp_path, FailingSim)
        terms = Terms()
        outcomes = []
        for _ in range(30):
            outcomes.append(await feeder.top_up(terms))
            await feeder.drain()
        assert outcomes == [TopUp.SUBMITTED] + [TopUp.IDLE] * 29
        assert sim.calls == 1 and len(feeder.runner.records) == 1 and terms.auto_refill is False

    asyncio.run(main())


def test_top_up_for_an_ibm_pool_is_a_programming_error(tmp_path):
    async def main():
        feeder, _ = feeder_for(tmp_path)
        terms = Terms(provider="ibm", implementation="fake_toronto", pool_key="ibm__fake_toronto__5")
        for force in (True, False):
            with pytest.raises(ValueError, match="explicit prefetch"):
                await feeder.top_up(terms, force=force)
        assert feeder.runner.records == {} and terms.auto_refill is True

    asyncio.run(main())


# --- landings ---------------------------------------------------------------------------------
@pytest.mark.parametrize("refill", [False, True])
def test_a_job_measured_on_another_device_leaves_the_pool_file_unchanged(tmp_path, refill):
    async def main():
        feeder, _ = feeder_for(tmp_path, ElsewhereSim)
        seed(feeder.store, 10)
        path = feeder.store.dir / "sim__aer_ideal__5.json"
        before = path.read_bytes()
        sub = feeder.bus.subscribe(None)
        terms = Terms()
        if refill:
            assert await feeder.top_up(terms) is TopUp.SUBMITTED
        else:
            await feeder.prefetch("sim", 5, 20)
        await feeder.drain()
        [error] = [d for e, d in drain_events(sub) if e == "error"]
        assert error["detail"].startswith("ValueError: job ran on other, pool is for aer_ideal")
        assert error["detail"].endswith(SUFFIX) is refill and terms.auto_refill is not refill
        assert path.read_bytes() == before and sorted(p.name for p in feeder.store.dir.glob("*.json")) == [path.name]

    asyncio.run(main())


def test_prefetch_returns_the_record_still_submitting_and_raises_only_before_recording(tmp_path):
    async def main():
        feeder, _ = feeder_for(tmp_path)
        placed = await feeder.prefetch("sim", 3, 20)
        assert placed.record.to_dict()["state"] == "submitting" and placed.pool_key == "sim__aer_ideal__3"
        await feeder.drain()
        assert feeder.store.available("sim__aer_ideal__3") == 20 and placed.record.state == "done"
        n = len(feeder.runner.records)
        for args, code in ((("nope", 3, 20), "unknown_provider"), (("sim", 0, 20), "invalid_num_qubits"),
                           (("sim", 3, 0), "invalid_shots")):
            with pytest.raises(AppError) as err:
                await feeder.prefetch(*args)
            assert err.value.code == code
        assert len(feeder.runner.records) == n

    asyncio.run(main())


def test_prefetch_resolve_failure_is_502_and_records_nothing(tmp_path):
    class Unresolvable(SimProvider):
        def resolve(self):
            raise RuntimeError("no device")

    async def main():
        feeder, _ = feeder_for(tmp_path, Unresolvable)
        with pytest.raises(AppError) as err:
            await feeder.prefetch("sim", 3, 20)
        assert (err.value.status, err.value.code, err.value.detail) == (502, "provider_error", "RuntimeError: no device")
        assert feeder.runner.records == {}

    asyncio.run(main())


def test_ibm_prefetch_ceiling_is_429_at_submit_after_resolve(tmp_path):
    async def main():
        settings = make_settings(tmp_path, ibm_token="TOK", ibm_max_total_shots=10)
        fake = FakeIbm()
        ibm = IbmProvider(settings, service_factory=fake.service_factory, sampler_cls=fake.sampler_cls,
                          poll_seconds=0.01)
        bus, budget = EventBus(), IbmBudget(settings)
        feeder = PoolFeeder({"ibm": ibm}, PoolStore(settings.data_dir), JobRunner(bus, budget, 0.0), bus, budget)
        with pytest.raises(AppError) as err:
            await feeder.prefetch("ibm", 3, 50)
        assert (err.value.status, err.value.code) == (429, "shot_ceiling")
        assert len(fake.service_calls) == 1 and fake.jobs == 0  # resolved first, as before; nothing submitted
        assert feeder.runner.records == {} and budget.used_jobs == 0

    asyncio.run(main())


def test_a_prefetch_resolving_when_shutdown_begins_is_503_and_submits_nothing(tmp_path):
    class GatedResolve(SimProvider):
        def __init__(self, settings):
            super().__init__(settings)
            self.entered = threading.Event()
            self.gate = threading.Event()

        def resolve(self):
            self.entered.set()
            assert self.gate.wait(10), "the test never opened the gate"
            return super().resolve()

    async def main():
        feeder, sim = feeder_for(tmp_path, GatedResolve)
        prefetch = asyncio.ensure_future(feeder.prefetch("sim", 3, 20))
        try:
            await until(sim.entered.is_set, "resolve to be running")
            feeder.runner.stop.set()  # what SessionManager.shutdown does first
            await feeder.close()
        finally:
            sim.gate.set()
        with pytest.raises(AppError) as err:
            await prefetch
        assert (err.value.status, err.value.code) == (503, "shutting_down")
        await feeder.drain()
        assert feeder.runner.records == {} and feeder.store.list() == []

    asyncio.run(main())


def test_close_cancels_the_landings(tmp_path):
    async def main():
        feeder, sim = feeder_for(tmp_path, GatedSim)
        sub = feeder.bus.subscribe(None)
        placed = await feeder.prefetch("sim", 3, 20)
        try:
            await until(sim.entered.is_set, "the job to be running")
            await feeder.close()  # the job is held by the gate: the landing is cancelled before it finishes
        finally:
            sim.gate.set()
        await until(lambda: placed.record.finished_at is not None, "the job to finish on the device")
        await asyncio.sleep(0.1)
        await feeder.drain()
        assert placed.record.discarded and feeder.store.list() == []  # a prefetch job is never adopted
        assert [e for e, _ in drain_events(sub) if e != "job"] == []

    asyncio.run(main())


# --- discarded live jobs ----------------------------------------------------------------------
async def _abandon(feeder, sim, kind="stream", before_release=lambda: None):
    """Cancel the caller of a running gated job, then let the job finish (cancel precedes completion)."""
    rec = feeder.runner.submit(sim, 3, 10, kind, "s1")
    task = asyncio.ensure_future(feeder.runner.execute(rec, sim))
    try:
        await until(sim.entered.is_set, "the job to be running")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        before_release()
    finally:
        sim.gate.set()
    return rec


def test_a_discarded_live_job_is_pooled_under_the_device_that_measured_it(tmp_path):
    async def main():
        feeder, sim = feeder_for(tmp_path, GatedElsewhereSim)
        sub = feeder.bus.subscribe(None)
        rec = await _abandon(feeder, sim)
        await until(lambda: rec.pooled_to is not None, "the bits to be pooled")
        await feeder.drain()
        events = drain_events(sub)
        assert [(e, d.get("pooled_to")) for e, d in events][-2:] == [("job", "sim__other__3"), ("pool", None)]
        assert events[-1][1]["job_id"] == rec.id and rec.pooled_to == "sim__other__3"
        assert feeder.store.available("sim__other__3") == 10 and feeder.store.get("sim__aer_ideal__3") is None

    asyncio.run(main())


@pytest.mark.parametrize("why", ["shutdown", "prefetch"])
def test_a_discarded_job_is_not_pooled_during_shutdown_or_when_it_is_a_prefetch(tmp_path, why):
    async def main():
        feeder, sim = feeder_for(tmp_path, GatedSim)
        stop = feeder.runner.stop.set if why == "shutdown" else lambda: None
        rec = await _abandon(feeder, sim, kind="prefetch" if why == "prefetch" else "stream", before_release=stop)
        await until(lambda: rec.finished_at is not None, "the job to finish")
        await asyncio.sleep(0.1)
        await feeder.drain()
        assert rec.discarded and rec.state == "done" and rec.pooled_to is None and feeder.store.list() == []

    asyncio.run(main())


# --- bank -------------------------------------------------------------------------------------
def test_bank_refuses_a_job_from_another_device_and_leaves_the_store_alone(tmp_path):
    store = PoolStore(tmp_path)
    seed(store, 10)
    path = store.dir / "sim__aer_ideal__5.json"
    before = path.read_bytes()
    with pytest.raises(ValueError) as err:
        bank(store, "sim", 5, measurement(5, 7, backend="other"), expect="aer_ideal")
    assert str(err.value) == "job ran on other, pool is for aer_ideal"
    assert path.read_bytes() == before and [p.key for p in store.list()] == ["sim__aer_ideal__5"]


def test_bank_round_trips_the_block(tmp_path):
    store = PoolStore(tmp_path)
    m = measurement(5, 7, backend="other", job_id="sim-salvaged", ones=True)
    pool = bank(store, "sim", 5, m, expect=None)
    assert pool.key == "sim__other__5" and pool.available == 7
    m2 = measurement(5, 3, job_id="sim-expected")
    assert bank(store, "sim", 5, m2, expect="aer_ideal").key == "sim__aer_ideal__5"
    [piece] = PoolStore(tmp_path).take("sim__other__5", 7)  # read back from disk by a fresh store
    assert (piece.job_id, piece.backend, piece.physical_qubits, piece.measured_at) == \
        ("sim-salvaged", "other", (0, 1, 2, 3, 4), m.measured_at)
    assert np.array_equal(piece.bits, m.bits)


# --- check_prefetch: the text the CLI and the API have always given -----------------------------
def providers_for(tmp_path, token):
    settings = make_settings(tmp_path, ibm_token=token, ibm_max_shots_per_job=2_000_000,
                             ibm_max_total_shots=3_000_000)
    fake = FakeIbm()
    ibm = IbmProvider(settings, service_factory=fake.service_factory, sampler_cls=fake.sampler_cls)
    return {"sim": SimProvider(settings), "ibm": ibm}, IbmBudget(settings), fake


@pytest.mark.parametrize("token, args, status, code, detail", [
    ("T", ("nope", 17, 0), 400, "unknown_provider", "unknown provider 'nope'; known: ['ibm', 'sim']"),
    (None, ("ibm", 17, 0), 409, "provider_unavailable", "provider ibm unavailable: IBM_QUANTUM_TOKEN not set"),
    ("T", ("sim", 17, 0), 400, "invalid_num_qubits", "num_qubits must be 1..16, got 17"),
    ("T", ("sim", 0, 10), 400, "invalid_num_qubits", "num_qubits must be 1..16, got 0"),
    ("T", ("ibm", 17, 0), 400, "invalid_num_qubits", "num_qubits must be 1..16, got 17"),
    ("T", ("sim", 5, 0), 400, "invalid_shots", "shots must be >= 1, got 0"),
    ("T", ("ibm", 5, 0), 400, "invalid_shots", "shots must be >= 1, got 0"),
    ("T", ("sim", 5, 1_000_001), 400, "invalid_shots", "shots must be 1..1000000, got 1000001"),
    ("T", ("ibm", 5, 2_000_001), 400, "shots_exceed_job_ceiling",
     "2000001 shots requested, IBM_MAX_SHOTS_PER_JOB is 2000000"),
])
def test_check_prefetch_refuses_with_the_existing_codes_and_text(tmp_path, token, args, status, code, detail):
    providers, budget, fake = providers_for(tmp_path, token)
    with pytest.raises(AppError) as err:
        check_prefetch(providers, *args, budget)
    assert (err.value.status, err.value.code, err.value.detail) == (status, code, detail)
    assert fake.service_calls == [] and (budget.used_shots, budget.used_jobs) == (0, 0)


def test_check_prefetch_passes_counts_nothing_and_caps_only_non_ibm_at_max_shots_per_job(tmp_path):
    providers, budget, fake = providers_for(tmp_path, "T")
    assert check_prefetch(providers, "sim", 5, 1_000_000, budget) is providers["sim"]
    assert check_prefetch(providers, "ibm", 16, 1_500_000, budget) is providers["ibm"]
    assert lookup_provider(providers, "ibm") is providers["ibm"]
    assert fake.service_calls == [] and (budget.used_shots, budget.used_jobs) == (0, 0)
