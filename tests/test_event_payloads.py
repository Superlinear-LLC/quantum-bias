"""Pool-feeding events as literal dicts: keys, key order, text, and the order the events come in.

Pinned before pool feeding moved into pool_feed.py; they pass unchanged after it. Everything is
driven through HTTP or the session manager's public methods, never its internals.
"""
import asyncio
import re
import threading
from datetime import datetime, timezone

import httpx
import numpy as np

from conftest import make_settings
from quantum_lite.app import create_app
from quantum_lite.errors import AppError
from quantum_lite.pools import PoolBlock
from quantum_lite.providers.base import Measurement
from quantum_lite.providers.sim import SimProvider

TS = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{6})?\+00:00"
REC_ID = r"[0-9a-f]{12}"
SIM_JOB = r"sim-[0-9a-f]{12}"
SUFFIX = " (auto_refill switched off for this session; no automatic retry)"
SESSION = {"provider": "sim", "mode": "buffered", "num_qubits": 5, "aggregation": "none",
           "shots_per_job": 1, "interval_ms": 20, "auto_refill": True, "refill_below": 50,
           "refill_shots": 50, "stream_seconds": 60, "stream_max_jobs": None}


class Like:
    """A string matching `pattern` in full (ids, timestamps)."""

    def __init__(self, pattern: str) -> None:
        self.pattern = re.compile(pattern)

    def __eq__(self, other) -> bool:
        return isinstance(other, str) and self.pattern.fullmatch(other) is not None

    def __repr__(self) -> str:
        return f"Like({self.pattern.pattern!r})"


class Number:
    def __eq__(self, other) -> bool:
        return isinstance(other, (int, float)) and not isinstance(other, bool)

    def __repr__(self) -> str:
        return "Number()"


def same(actual, expected, path="$"):
    """Equal, and every dict has the same keys in the same order."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: {actual!r} is not a dict"
        assert list(actual) == list(expected), f"{path}: keys {list(actual)} != {list(expected)}"
        for k in expected:
            same(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: {actual!r} != {expected!r}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, f"{path}[{i}]")
    else:
        assert expected == actual, f"{path}: {actual!r} != {expected!r}"
        if not isinstance(expected, (Like, Number)):  # True == 1, but a bool is not a count
            assert type(actual) is type(expected), f"{path}: {type(actual).__name__} is not {type(expected).__name__}"


class Events:
    """Everything one subscription received, in order."""

    def __init__(self, sub) -> None:
        self.sub = sub
        self.all: list[tuple[str, dict]] = []

    def pump(self) -> list[tuple[str, dict]]:
        while not self.sub.queue.empty():
            self.all.append(self.sub.queue.get_nowait())
        return self.all

    def of(self, *names: str) -> list[tuple[str, dict]]:
        return [(e, d) for e, d in self.pump() if e in names]

    async def until(self, predicate, what: str, timeout: float = 10.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(f"timed out waiting for {what}; events: {[e for e, _ in self.pump()]}")
            await asyncio.sleep(0.01)


def http(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def seed(mgr, shots, n=5):
    mgr.pools.append("sim", "aer_ideal", n, PoolBlock("seed", "aer_ideal", tuple(range(n)),
                                                     "2026-01-01T00:00:00+00:00", np.zeros((shots, n), np.uint8)))


class FailingSim(SimProvider):
    def run(self, n, shots, on_state, stop):
        raise RuntimeError("device error")


class GatedSim(SimProvider):
    """Every job, once running, waits for `gate` before it measures."""

    def __init__(self, settings):
        super().__init__(settings)
        self.gate = threading.Event()

    def execute(self, circuit, n, shots):
        assert self.gate.wait(10), "the test never opened the gate"
        return super().execute(circuit, n, shots)


class ElsewhereSim(SimProvider):
    """Resolves to aer_ideal, measures on another device."""

    def run(self, n, shots, on_state, stop):
        return Measurement(np.zeros((shots, n), np.uint8), "sim-elsewhere", "sim", "other", tuple(range(n)),
                           datetime.now(timezone.utc).isoformat())


def job(**kw) -> dict:
    """A job event payload: JobRecord.to_dict() plus ts (session_id keeps its place)."""
    base = {"id": Like(REC_ID), "job_id": None, "provider": "sim", "backend": None, "kind": "prefetch",
            "shots": 20, "n_qubits": 3, "session_id": None, "state": "submitting", "queue_position": None,
            "submitted_at": Like(TS), "finished_at": None, "error": None, "simulated_latency": False,
            "discarded": False, "pooled_to": None, "estimated_qpu_seconds": None, "estimate_formula": None}
    base.update(kw)
    return base


def pool(key, n, blocks, consumed=0) -> dict:
    total = sum(b["shots"] for b in blocks)
    provider, implementation, _ = key.split("__")
    return {"key": key, "provider": provider, "implementation": implementation, "n_qubits": n,
            "total_shots": total, "consumed_shots": consumed, "available_shots": total - consumed,
            "blocks": blocks}


# --- explicit prefetch ------------------------------------------------------------------------
def test_prefetch_202_body_then_job_states_then_pool(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        ev = Events(mgr.bus.subscribe(None))
        async with http(app) as c:
            r = await c.post("/api/pools/prefetch", json={"provider": "sim", "num_qubits": 3, "shots": 20})
        assert r.status_code == 202, r.text
        body = r.json()
        same(body, {"job": job(), "pool_key": "sim__aer_ideal__3"})
        rid = body["job"]["id"]
        await ev.until(lambda: ev.of("pool"), "the pool event")
        assert [(e, d.get("state")) for e, d in ev.all] == [
            ("job", "submitting"), ("job", "queued"), ("job", "running"), ("job", "done"), ("pool", None)]
        done = ev.all[3][1]
        same(done, {**job(id=rid, job_id=Like(SIM_JOB), backend="aer_ideal", state="done",
                          finished_at=Like(TS)), "ts": Like(TS)})
        same(ev.all[4][1], {
            "pool": pool("sim__aer_ideal__3", 3, [{"job_id": done["job_id"], "backend": "aer_ideal", "shots": 20,
                                                   "measured_at": Like(TS)}]),
            "job_id": rid, "ts": Like(TS)})
        await mgr.shutdown()

    asyncio.run(main())


def test_failed_prefetch_landing_is_a_global_error_event(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        app = create_app(settings, providers={"sim": FailingSim(settings)})
        mgr = app.state.mgr
        ev = Events(mgr.bus.subscribe(None))
        async with http(app) as c:
            r = await c.post("/api/pools/prefetch", json={"provider": "sim", "num_qubits": 3, "shots": 20})
        rid = r.json()["job"]["id"]
        await ev.until(lambda: ev.of("error"), "the error event")
        assert [(e, d.get("state")) for e, d in ev.all] == [("job", "submitting"), ("job", "error"), ("error", None)]
        same(ev.all[2][1], {"scope": "prefetch", "pool_key": "sim__aer_ideal__3", "job_id": rid,
                            "error": "prefetch_error", "detail": "RuntimeError: device error", "ts": Like(TS)})
        assert mgr.pools.list() == []
        await mgr.shutdown()

    asyncio.run(main())


def test_prefetch_measured_on_another_device_is_a_global_error_event(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        app = create_app(settings, providers={"sim": ElsewhereSim(settings)})
        mgr = app.state.mgr
        ev = Events(mgr.bus.subscribe(None))
        async with http(app) as c:
            r = await c.post("/api/pools/prefetch", json={"provider": "sim", "num_qubits": 3, "shots": 20})
        rid = r.json()["job"]["id"]
        await ev.until(lambda: ev.of("error"), "the error event")
        assert [(e, d.get("state")) for e, d in ev.all] == [("job", "submitting"), ("job", "done"), ("error", None)]
        same(ev.all[2][1], {"scope": "prefetch", "pool_key": "sim__aer_ideal__3", "job_id": rid,
                            "error": "prefetch_error", "detail": "ValueError: job ran on other, pool is for aer_ideal",
                            "ts": Like(TS)})
        assert mgr.pools.list() == []
        await mgr.shutdown()

    asyncio.run(main())


# --- auto_refill --------------------------------------------------------------------------------
def test_failed_refill_landing_is_a_session_error_with_the_suffix(tmp_path):
    async def main():
        settings = make_settings(tmp_path)
        app = create_app(settings, providers={"sim": FailingSim(settings)})
        mgr = app.state.mgr
        s = await mgr.create(SESSION)
        seed(mgr, 10)
        ev = Events(mgr.bus.subscribe(s.id))
        await mgr.press(s)
        await ev.until(lambda: ev.of("error"), "the refill's error event")
        jobs_and_errors = ev.of("job", "error")
        assert [(e, d.get("state")) for e, d in jobs_and_errors] == [
            ("job", "submitting"), ("job", "error"), ("error", None)]
        rid = jobs_and_errors[0][1]["id"]
        same(jobs_and_errors[0][1], {**job(id=rid, shots=50, n_qubits=5, session_id=s.id), "ts": Like(TS)})
        same(jobs_and_errors[2][1], {"scope": "prefetch", "pool_key": "sim__aer_ideal__5", "job_id": rid,
                                     "error": "prefetch_error", "detail": "RuntimeError: device error" + SUFFIX,
                                     "ts": Like(TS), "session_id": s.id})
        assert s.auto_refill is False
        await mgr.shutdown()

    asyncio.run(main())


def test_refused_refill_submit_is_a_refill_scope_error_before_the_bit(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        s = await mgr.create(SESSION)
        seed(mgr, 10)
        real = mgr.runner.submit

        def submit(provider, n, shots, kind, session_id):
            if kind == "prefetch":
                raise AppError(429, "shot_ceiling", "IBM_MAX_TOTAL_SHOTS is 10: 0 used + 50 requested")
            return real(provider, n, shots, kind, session_id)

        mgr.runner.submit = submit
        ev = Events(mgr.bus.subscribe(s.id))
        await mgr.press(s)
        assert [e for e, _ in ev.pump()] == ["error", "bit", "stats"]
        same(ev.all[0][1], {"scope": "refill", "pool_key": "sim__aer_ideal__5", "error": "shot_ceiling",
                            "detail": "IBM_MAX_TOTAL_SHOTS is 10: 0 used + 50 requested" + SUFFIX,
                            "ts": Like(TS), "session_id": s.id})
        assert s.auto_refill is False and mgr.runner.records == {}
        await mgr.press(s)
        assert [e for e, _ in ev.pump()][3:] == ["bit", "stats"]  # off: no second attempt, no second error
        await mgr.shutdown()

    asyncio.run(main())


def test_stream_caused_refill_publishes_job_then_stream_then_pool(tmp_path):
    async def main():
        app = create_app(make_settings(tmp_path))
        mgr = app.state.mgr
        s = await mgr.create({**SESSION, "shots_per_job": 10, "refill_below": 80, "refill_shots": 100,
                              "stream_max_jobs": 10})
        seed(mgr, 50)
        ev = Events(mgr.bus.subscribe(s.id))
        glob = Events(mgr.bus.subscribe(None))
        await mgr.start_stream(s)
        await glob.until(lambda: any("job_id" in d for _, d in glob.of("pool")), "the refill's pool event")
        await mgr.stop_stream(s)
        first = ev.pump()[:3]
        assert [e for e, _ in first] == ["stream", "job", "stream"]
        stream = {"running": True, "error": None, "stopped_reason": None, "seconds_limit": 60,
                  "seconds_left": Number(), "jobs_limit": 10}
        same(first[0][1], {"stream": {**stream, "jobs_used": 0}, "ts": Like(TS), "session_id": s.id})
        rid = first[1][1]["id"]
        same(first[1][1], {**job(id=rid, shots=100, n_qubits=5, session_id=s.id), "ts": Like(TS)})
        same(first[2][1], {"stream": {**stream, "jobs_used": 1}, "ts": Like(TS), "session_id": s.id})
        # The whole run, not just its start: `stream` is published at start and right after each refill
        # submit the run counted (stop_stream publishes none). A `stream` after a top_up that submitted
        # nothing (idle, busy) has no `submitting` job right before it.
        names = [e for e, _ in ev.all]
        streams = [i for i, e in enumerate(names) if e == "stream"]
        submits = [i for i, (e, d) in enumerate(ev.all) if e == "job" and d["state"] == "submitting"]
        assert streams == [0] + [i + 1 for i in submits], names
        assert [ev.all[i][1]["stream"]["jobs_used"] for i in streams] == list(range(len(streams))), names
        assert all(ev.all[i][1]["kind"] == "prefetch" for i in submits), names
        landed = [d for _, d in glob.of("pool") if "job_id" in d]
        assert len(landed) == 1
        same(landed[0], {"pool": {**landed[0]["pool"]}, "job_id": rid, "ts": Like(TS)})
        assert landed[0]["pool"]["blocks"][-1]["shots"] == 100
        await mgr.shutdown()

    asyncio.run(main())


# --- a live job whose caller went away ---------------------------------------------------------
async def _orphan(tmp_path, fail_write: bool):
    settings = make_settings(tmp_path, sim_latency_seconds=0.3)  # latency only so the record says simulated_latency
    sim = GatedSim(settings)
    app = create_app(settings, providers={"sim": sim})
    mgr = app.state.mgr
    s = await mgr.create({**SESSION, "mode": "live", "num_qubits": 3, "shots_per_job": 10, "auto_refill": False,
                          "refill_below": 0, "refill_shots": None})
    if fail_write:
        def append(*a, **k):
            raise OSError("disk full")

        mgr.pools.append = append
    glob = Events(mgr.bus.subscribe(None))
    await mgr.start_stream(s)
    try:
        await glob.until(lambda: mgr.runner.has_inflight(s.id, "stream"), "the live job")
        await mgr.stop_stream(s)  # the job is held by the gate: the stream is cancelled before it finishes
    finally:
        sim.gate.set()
    return mgr, s, glob


def test_discarded_live_job_is_pooled_job_event_then_pool_event(tmp_path):
    async def main():
        mgr, s, glob = await _orphan(tmp_path, fail_write=False)
        await glob.until(lambda: glob.of("pool"), "the pool event")
        await asyncio.sleep(0.1)
        rid = glob.all[0][1]["id"]
        assert [(e, d.get("state"), d.get("pooled_to")) for e, d in glob.pump()] == [
            ("job", "submitting", None), ("job", "queued", None), ("job", "running", None),
            ("job", "done", None), ("job", "done", "sim__aer_ideal__3"), ("pool", None, None)]
        same(glob.all[4][1], {**job(id=rid, job_id=Like(SIM_JOB), backend="aer_ideal", kind="stream", shots=10,
                                    session_id=s.id, state="done", finished_at=Like(TS), simulated_latency=True,
                                    discarded=True, pooled_to="sim__aer_ideal__3"), "ts": Like(TS)})
        same(glob.all[5][1], {
            "pool": pool("sim__aer_ideal__3", 3, [{"job_id": glob.all[4][1]["job_id"], "backend": "aer_ideal",
                                                   "shots": 10, "measured_at": Like(TS)}]),
            "job_id": rid, "ts": Like(TS)})
        await mgr.shutdown()

    asyncio.run(main())


def test_discarded_live_job_that_cannot_be_written_is_a_discarded_job_error(tmp_path):
    async def main():
        mgr, s, glob = await _orphan(tmp_path, fail_write=True)
        await glob.until(lambda: glob.of("error"), "the error event")
        await asyncio.sleep(0.1)
        rid = glob.all[0][1]["id"]
        assert [(e, d.get("state"), d.get("pooled_to")) for e, d in glob.pump()] == [
            ("job", "submitting", None), ("job", "queued", None), ("job", "running", None),
            ("job", "done", None), ("error", None, None)]
        same(glob.all[4][1], {"scope": "discarded_job", "job_id": rid, "error": "pool_write_failed",
                              "detail": "OSError: disk full", "ts": Like(TS), "session_id": s.id})
        assert mgr.runner.records[rid].pooled_to is None
        await mgr.shutdown()

    asyncio.run(main())
