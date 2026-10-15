"""Sessions, mode segments, the interactions and the stream loop. Bits come from acquisition.py."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from quantum_lite.acquisition import MODES, Acquisition, Batch, CycleInterrupted, Stop
from quantum_lite.aggregation import validate_aggregation
from quantum_lite.budget import IbmBudget
from quantum_lite.circuit import validate_qubits
from quantum_lite.errors import AppError, off_loop
from quantum_lite.events import EventBus, now_iso
from quantum_lite.jobs import MAX_SHOTS_PER_JOB, JobRunner
from quantum_lite.pool_feed import PoolFeeder, lookup_provider
from quantum_lite.pools import PoolStore, pool_key
from quantum_lite.providers.base import Provider
from quantum_lite.stats import Stats
from quantum_lite.stream_run import StreamRun

log = logging.getLogger("quantum_lite.sessions")

MIN_INTERVAL_MS = 10
MAX_STREAM_SECONDS = 3600  # one stream runs at most an hour; start it again to go on
MAX_STREAM_JOBS = 1000


def tail_run(aggregated: np.ndarray) -> dict:
    """The run at the end of a batch of aggregated bits: its bit value and length.

    The UI counts streaks of consecutive bits equal to the target across batches. A batch can
    hold up to MAX_SHOTS_PER_JOB bits, so the event carries the trailing run, not the bits:
    `tail_run == shots` means the whole batch is one run and the streak continues through it.
    """
    n = int(aggregated.shape[0])
    if n == 0:
        raise ValueError("tail_run of an empty batch")
    last = int(aggregated[-1])
    differs = np.flatnonzero(aggregated != last)
    run = n if differs.size == 0 else n - 1 - int(differs[-1])
    return {"tail_bit": last, "tail_run": run}


@dataclass
class Segment:
    index: int
    mode: str
    stats: Stats
    started_at: str = field(default_factory=now_iso)
    ended_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index, "mode": self.mode, "started_at": self.started_at,
            "ended_at": self.ended_at, "stats": self.stats.snapshot(),
        }


class Session:
    def __init__(self, *, provider: str, implementation: str, mode: str, n_qubits: int, aggregation: str,
                 shots_per_job: int, interval_ms: int, auto_refill: bool, refill_below: int,
                 refill_shots: int, physical_qubits: list[int], stream_seconds: int,
                 stream_max_jobs: int | None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.provider = provider
        self.implementation = implementation
        self.mode = mode
        self.n_qubits = n_qubits
        self.aggregation = aggregation
        self.shots_per_job = shots_per_job
        self.interval_ms = interval_ms
        self.auto_refill = auto_refill
        self.refill_below = refill_below
        self.refill_shots = refill_shots
        self.physical_qubits = physical_qubits
        self.stream_seconds = stream_seconds
        self.stream_max_jobs = stream_max_jobs
        self.pool_key = pool_key(provider, implementation, n_qubits)
        self.created_at = now_iso()
        self.stats = Stats(n_qubits)
        self.mode_stats = {m: Stats(n_qubits) for m in MODES}  # all segments of one mode, combined
        self.segments: list[Segment] = [Segment(0, mode, Stats(n_qubits))]
        self.stream_task: asyncio.Task | None = None
        self.run: StreamRun | None = None  # the current (or last) stream run; None until the first start
        self.pending: dict | None = None  # perceive: prepared, not yet revealed
        self.preparing = False

    @property
    def segment(self) -> Segment:
        return self.segments[-1]

    @property
    def streaming(self) -> bool:
        return self.stream_task is not None and not self.stream_task.done()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "provider": self.provider, "implementation": self.implementation,
            "mode": self.mode, "num_qubits": self.n_qubits, "aggregation": self.aggregation,
            "shots_per_job": self.shots_per_job, "interval_ms": self.interval_ms,
            "auto_refill": self.auto_refill, "refill_below": self.refill_below,
            "refill_shots": self.refill_shots, "physical_qubits": self.physical_qubits,
            "stream_seconds": self.stream_seconds, "stream_max_jobs": self.stream_max_jobs,
            "pool_key": self.pool_key, "created_at": self.created_at,
            "stream": self.stream_dict(),
            "perceive_pending": self.pending is not None or self.preparing,
            "stats": self.stats_dict(),
        }

    def stream_dict(self, running: bool | None = None) -> dict:
        running = self.streaming if running is None else running
        if self.run is not None:
            return self.run.snapshot(running)
        return {
            "running": running, "error": None, "stopped_reason": None,
            "seconds_limit": self.stream_seconds, "seconds_left": None,
            "jobs_limit": self.stream_max_jobs, "jobs_used": 0,
        }

    def stats_dict(self) -> dict:
        return {
            "total": self.stats.snapshot(),
            "by_mode": {m: st.snapshot() for m, st in self.mode_stats.items()},
            "segments": [s.to_dict() for s in self.segments],
        }

    def scopes(self, segment: "Segment") -> tuple[Stats, Stats, Stats]:
        """Every Stats a bit of this segment counts toward: segment, its mode, the session."""
        return segment.stats, self.mode_stats[segment.mode], self.stats


class SessionManager:
    def __init__(self, providers: dict[str, Provider], runner: JobRunner, pools: PoolStore,
                 bus: EventBus, budget: IbmBudget, feeder: PoolFeeder, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.providers = providers
        self.runner = runner
        self.pools = pools  # the same PoolStore the feeder fills
        self.bus = bus
        self.budget = budget
        self.feeder = feeder
        self.clock = clock  # monotonic seconds; every stream run's deadline is on it
        self.acq = Acquisition(providers, runner, pools, feeder, bus)
        self.sessions: dict[str, Session] = {}

    # --- lookup / validation ---------------------------------------------------------
    def get(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise AppError(404, "session_not_found", f"no session {session_id}") from None

    def provider(self, provider_id: str) -> Provider:
        return lookup_provider(self.providers, provider_id)

    # --- create / switch -------------------------------------------------------------
    async def create(self, p: dict) -> Session:
        provider = self.provider(p["provider"])
        mode = p["mode"]
        if mode not in MODES:
            raise AppError(400, "invalid_mode", f"mode must be one of {MODES}, got {mode!r}")
        n = p["num_qubits"]
        validate_qubits(n)
        validate_aggregation(p["aggregation"], n)
        shots = p["shots_per_job"]
        if not (1 <= shots <= MAX_SHOTS_PER_JOB):
            raise AppError(400, "invalid_shots", f"shots_per_job must be 1..{MAX_SHOTS_PER_JOB}, got {shots}")
        if provider.id == "ibm":
            self.budget.check_job_size(shots)
        if p["interval_ms"] < MIN_INTERVAL_MS:
            raise AppError(400, "invalid_interval", f"interval_ms must be >= {MIN_INTERVAL_MS}, got {p['interval_ms']}")
        if provider.id == "ibm" and p["auto_refill"]:
            raise AppError(400, "auto_refill_ibm",
                           "auto_refill is not available on ibm: every refill there is a billed QPU job, "
                           "so ibm pools are refilled only by an explicit prefetch (POST /api/pools/prefetch)")
        if p["auto_refill"] and p["refill_below"] < 1:
            raise AppError(400, "invalid_refill", "auto_refill requires refill_below >= 1")
        if p["refill_below"] < 0:
            raise AppError(400, "invalid_refill", f"refill_below must be >= 0, got {p['refill_below']}")
        self._validate_stream_limits(provider.id, p["stream_seconds"], p["stream_max_jobs"])
        refill_shots = p["refill_shots"] if p["refill_shots"] is not None else max(p["refill_below"], shots)
        if refill_shots < 1:
            raise AppError(400, "invalid_refill", f"refill_shots must be >= 1, got {refill_shots}")
        if p["auto_refill"]:
            if refill_shots < p["refill_below"]:
                raise AppError(400, "invalid_refill",
                               f"refill_shots ({refill_shots}) must be >= refill_below ({p['refill_below']}): "
                               "a smaller refill cannot lift the pool above the threshold")
            if provider.id == "ibm":
                self.budget.check_job_size(refill_shots)
        implementation = await off_loop(provider.resolve)
        physical = await off_loop(provider.physical_qubits, n)
        s = Session(
            provider=provider.id, implementation=implementation, mode=mode, n_qubits=n,
            aggregation=p["aggregation"], shots_per_job=shots, interval_ms=p["interval_ms"],
            auto_refill=p["auto_refill"], refill_below=p["refill_below"], refill_shots=refill_shots,
            physical_qubits=physical, stream_seconds=p["stream_seconds"], stream_max_jobs=p["stream_max_jobs"],
        )
        self.sessions[s.id] = s
        return s

    @staticmethod
    def _validate_stream_limits(provider_id: str, seconds: int | None, max_jobs: int | None) -> None:
        # Every session can stream (the API starts one on any session), so every session needs a
        # duration. On ibm each live cycle is a billed QPU job, so ibm also needs a job count.
        if seconds is None:
            raise AppError(400, "invalid_stream_seconds",
                           f"stream_seconds is required (1..{MAX_STREAM_SECONDS}): how long a stream may run")
        if not (1 <= seconds <= MAX_STREAM_SECONDS):
            raise AppError(400, "invalid_stream_seconds",
                           f"stream_seconds must be 1..{MAX_STREAM_SECONDS}, got {seconds}")
        if max_jobs is None:
            if provider_id == "ibm":
                raise AppError(400, "invalid_stream_max_jobs",
                               "stream_max_jobs is required for ibm sessions: every live stream cycle "
                               "is a billed QPU job, so a stream needs a job limit")
            return
        if not (1 <= max_jobs <= MAX_STREAM_JOBS):
            raise AppError(400, "invalid_stream_max_jobs",
                           f"stream_max_jobs must be 1..{MAX_STREAM_JOBS}, got {max_jobs}")

    def switch_mode(self, s: Session, mode: str) -> Session:
        if mode not in MODES:
            raise AppError(400, "invalid_mode", f"mode must be one of {MODES}, got {mode!r}")
        if mode == s.mode:
            raise AppError(400, "mode_unchanged", f"session is already {mode}")
        s.segment.ended_at = now_iso()
        s.mode = mode
        s.segments.append(Segment(len(s.segments), mode, Stats(s.n_qubits)))
        self.bus.publish("stats", {"stats": s.stats_dict(), "reason": "mode_switch"}, s.id)
        return s

    # --- recording ------------------------------------------------------------------
    def record(self, s: Session, batch: Batch) -> None:
        for st in s.scopes(batch.segment):
            st.add_block(batch.bits, batch.aggregated)

    def _publish_stats(self, s: Session) -> None:
        self.bus.publish("stats", {"stats": s.stats_dict()}, s.id)

    # --- interactions ----------------------------------------------------------------
    async def press(self, s: Session) -> dict:
        batch = await self.acq.one(s, "press")
        self.record(s, batch)
        bit = int(batch.aggregated[0])
        self.bus.publish("bit", {"kind": "press", "bit": bit, "provenance": batch.provenance}, s.id)
        self._publish_stats(s)
        return {"bit": bit, "provenance": batch.provenance, "stats": s.stats_dict()}

    async def predict(self, s: Session, guess: int) -> dict:
        # The guess is in hand before acquisition starts. Live: the job is submitted after it.
        # Buffered: the bit was measured long before; flagged below, never hidden.
        batch = await self.acq.one(s, "predict")
        self.record(s, batch)
        bit = int(batch.aggregated[0])
        hit = bit == guess
        for st in s.scopes(batch.segment):
            st.predict.add(hit)
        measured_before_guess = batch.provenance["source"] == "buffered"
        result = {"bit": bit, "guess": guess, "hit": hit, "measured_before_guess": measured_before_guess,
                  "provenance": batch.provenance}
        self.bus.publish("bit", {"kind": "predict", **result}, s.id)
        self._publish_stats(s)
        return {**result, "stats": s.stats_dict()}

    async def perceive_prepare(self, s: Session) -> dict:
        if s.pending is not None or s.preparing:
            raise AppError(409, "perceive_pending", "a perceive trial is already prepared; reveal it first")
        s.preparing = True
        try:
            batch = await self.acq.one(s, "perceive")
        finally:
            s.preparing = False
        trial_id = uuid.uuid4().hex[:12]
        # The bit stays server-side. It is not counted in stats until reveal, so the stats
        # events cannot leak it.
        s.pending = {"trial_id": trial_id, "batch": batch}
        return {"trial_id": trial_id, "source": batch.provenance["source"]}

    async def perceive_reveal(self, s: Session, trial_id: str, guess: int) -> dict:
        pending = s.pending
        if pending is None or pending["trial_id"] != trial_id:
            raise AppError(404, "unknown_trial", f"no prepared perceive trial {trial_id!r} in session {s.id}")
        s.pending = None
        batch: Batch = pending["batch"]
        self.record(s, batch)
        bit = int(batch.aggregated[0])
        hit = bit == guess
        for st in s.scopes(batch.segment):
            st.perceive.add(hit)
        result = {"trial_id": trial_id, "bit": bit, "guess": guess, "hit": hit, "provenance": batch.provenance}
        self.bus.publish("bit", {"kind": "perceive", **result}, s.id)
        self._publish_stats(s)
        return {**result, "stats": s.stats_dict()}

    # --- stream ----------------------------------------------------------------------
    async def start_stream(self, s: Session) -> None:
        if s.streaming:
            raise AppError(409, "stream_running", "stream already running")
        await self.acq.admit_stream(s)  # from here to create_task there is no await
        s.run = StreamRun(s.stream_seconds, s.stream_max_jobs, self.clock)
        s.stream_task = asyncio.create_task(self._stream_loop(s, s.run))
        self.bus.publish("stream", {"stream": s.stream_dict()}, s.id)

    async def stop_stream(self, s: Session) -> None:
        if not s.streaming:
            raise AppError(409, "stream_not_running", "stream is not running")
        task = s.stream_task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # we cancelled it ourselves; the loop never re-raises anything else

    def listener_gone(self, s: Session) -> None:
        """An event listener of the session disconnected; stop its stream if it was the last."""
        if s.streaming and not self.bus.has_subscribers(s.id):
            s.run.end_by_error("stream_stopped: the last event listener disconnected")
            s.stream_task.cancel()
            log.warning("stream %s stopped: no event listener left", s.id)

    def _limit_stop(self, s: Session, code: str, detail: str) -> None:
        """A stream limit ended the run (the run already recorded it). Not an error: the reason is in an event."""
        log.info("stream %s stopped: %s: %s", s.id, code, detail)
        self.bus.publish("stream_stopped", {"code": code, "detail": detail, "stream": s.stream_dict(running=False)}, s.id)

    async def _stream_loop(self, s: Session, run: StreamRun) -> None:
        try:
            while True:
                t0 = self.clock()
                if run.expired():
                    self._limit_stop(s, *run.end_by_deadline())
                    return
                # One capture of the segment decides the cycle (Acquisition.cycle); a mode switch
                # takes effect at the next cycle. Live: when no stream job of the session is in
                # flight, the permit is reached with no suspension; otherwise the cycle first waits
                # for that job (bounded by the deadline), then captures. The run's limits are judged
                # at the submit itself either way.
                c = await self.acq.cycle(s, run)
                if c.batch is not None:
                    self._deliver(s, c.batch)
                if c.stop is Stop.REFUSED:
                    self._limit_stop(s, *run.end_by_refusal())
                    return
                if c.stop is Stop.ABANDONED:
                    self._limit_stop(s, *run.end_by_deadline(job_in_flight=True))
                    return
                if c.stop is Stop.EXPIRED:  # nothing of this cycle is in flight
                    self._limit_stop(s, *run.end_by_deadline())
                    return
                await c.pace(t0)
        except CycleInterrupted as stopped:  # stopped after taking pool shots: they are recorded
            self._deliver(s, stopped.batch)
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, AppError) else "stream_error"
            run.end_by_error(f"{code}: {exc}")
            log.error("stream %s stopped: %s: %s", s.id, code, exc)
            self.bus.publish("error", {"scope": "stream", "error": code, "detail": str(exc)}, s.id)

    def _deliver(self, s: Session, batch: Batch) -> None:
        self.record(s, batch)
        self.bus.publish("batch", {
            "shots": int(batch.bits.shape[0]),
            "raw_ones": int(batch.bits.sum(dtype=np.int64)),
            "aggregated_ones": int(batch.aggregated.sum(dtype=np.int64)),
            **tail_run(batch.aggregated),
            "provenance": batch.provenance,
        }, s.id)
        self._publish_stats(s)

    async def shutdown(self) -> None:
        self.runner.stop.set()
        tasks = [s.stream_task for s in self.sessions.values() if s.streaming]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, self.feeder.close(), return_exceptions=True)
