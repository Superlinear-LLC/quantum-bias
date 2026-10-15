"""Acquisition: getting the bits for one interaction or one stream cycle (CONTEXT.md).

Every live/buffered rule is here, in two adjacent sources:
- live: a fresh job per acquisition. A stream cycle's job goes through the permit (StreamRun.submit),
  which checks, submits and counts in one synchronous step, and is given up at the run's deadline:
  it cannot be recalled, it finishes and the pool feeder keeps its bits. Pipeline depth 1 is kept
  at the permit: a cycle that finds the session's previous stream job still running (a stopped
  run's, say) waits for it before its capture, so a session never has two stream jobs in flight.
- buffered: the next shots of the session's pool. A take is never given up (that would lose pool
  shots): not at the deadline, and not when the stream is stopped (CycleInterrupted carries its
  batch out of the cancelled cycle to be recorded). Its refill goes through the permit when a
  stream caused it. Paced by interval_ms.

One capture decides an acquisition: it reads the session's current segment, and that segment alone
gives the source, the provenance (source, segment) and, for a cycle, whether it is given up at the
deadline and how it is paced. A mode switch while bits are in flight cannot move them. For `one`
the capture is the first step. For a cycle it follows the depth-1 wait, if there is one, and is
taken afresh after it, so a switch during the wait is obeyed (no live job after a switch away from
live); from the capture to the permit there is no await. The source comes only from the segment's
mode: nothing is retried, padded or redirected.

This module never imports sessions; a Session satisfies AcquireTerms.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Literal, Mapping, Protocol

import numpy as np

from quantum_lite.aggregation import aggregate
from quantum_lite.errors import AppError, provider_failure
from quantum_lite.events import EventBus
from quantum_lite.jobs import JobRecord, JobRunner
from quantum_lite.pool_feed import PoolFeeder, RefillTerms, TopUp
from quantum_lite.pools import PoolBlock, PoolStore
from quantum_lite.providers.base import Provider
from quantum_lite.stream_run import StreamRefused, StreamRun

log = logging.getLogger("quantum_lite.acquisition")

MODES = ("live", "buffered")
IN_FLIGHT_DETAIL = "the previous stream job is still in flight; wait for it to finish"


class SegmentRef(Protocol):
    index: int
    mode: str


class AcquireTerms(RefillTerms, Protocol):
    """What an acquisition needs from a session. Session satisfies it."""

    mode: str
    aggregation: str
    shots_per_job: int
    interval_ms: int

    @property
    def segment(self) -> SegmentRef: ...

    @property
    def streaming(self) -> bool: ...

    def stream_dict(self) -> dict: ...


@dataclass
class Batch:
    bits: np.ndarray  # (shots, n)
    aggregated: np.ndarray  # (shots,)
    provenance: dict
    segment: SegmentRef  # the segment the acquisition started in; its stats count these bits


class Stop(Enum):
    REFUSED = "refused"  # a stream limit refused a submit; the refusal is pending on the run
    ABANDONED = "abandoned"  # a live job given up at the deadline; it still runs and will be pooled
    EXPIRED = "expired"  # the deadline passed while the cycle waited for the previous stream job; nothing submitted


class CycleInterrupted(asyncio.CancelledError):
    """The stream was stopped (cancelled) after its cycle had taken shots from the pool.

    Those shots are consumed and cannot be put back, so the cycle finished taking them and carries
    them out: the stream loop records and publishes `batch`, then lets the cancellation end it.
    """

    def __init__(self, batch: "Batch | None" = None, blocks: "Blocks | None" = None) -> None:
        super().__init__("stream stopped after its cycle took shots from the pool")
        self.batch = batch
        self.blocks = blocks


async def _no_pace(t0: float) -> None:
    return None


@dataclass(frozen=True)
class Cycle:
    """One stream cycle's outcome: the batch it delivered, and the stop it ran into (either may be None, not both)."""

    batch: Batch | None
    stop: Stop | None
    _pace: Callable[[float], Awaitable[None]] = field(default=_no_pace, repr=False, compare=False)

    async def pace(self, t0: float) -> None:
        """Wait out the rest of this cycle; t0 is the loop's clock read at the top of the cycle."""
        await self._pace(t0)


Blocks = list[PoolBlock]


class _Source(Protocol):
    def busy(self, s: AcquireTerms) -> bool: ...

    async def settle(self, s: AcquireTerms, run: StreamRun) -> bool: ...

    async def ready(self, s: AcquireTerms) -> None: ...

    async def one(self, s: AcquireTerms, kind: str) -> Blocks: ...

    async def cycle(self, s: AcquireTerms, run: StreamRun) -> tuple[Blocks | None, Stop | None]: ...

    async def pace(self, s: AcquireTerms, run: StreamRun, t0: float) -> None: ...


class _LiveSource:
    def __init__(self, providers: Mapping[str, Provider], runner: JobRunner, bus: EventBus) -> None:
        self._providers = providers
        self._runner = runner
        self._bus = bus

    def busy(self, s: AcquireTerms) -> bool:
        """Depth 1: a stream job of the session is still in flight (a stopped run's; it cannot be recalled)."""
        return self._runner.has_inflight(s.id, "stream")

    async def settle(self, s: AcquireTerms, run: StreamRun) -> bool:
        """Wait for that job, at most until the deadline. True when it has finished."""
        return await self._runner.settled(s.id, "stream", run.seconds_left())

    async def ready(self, s: AcquireTerms) -> None:
        return None  # nothing to check before a live stream starts; returns without suspending

    async def one(self, s: AcquireTerms, kind: str) -> Blocks:
        provider = self._providers[s.provider]
        try:
            m = await self._runner.run(provider, s.n_qubits, 1, kind, s.id)
        except AppError:
            raise  # a ceiling refused the submit (400 / 429)
        except Exception as exc:
            raise provider_failure(exc) from exc
        return [PoolBlock.of(m)]

    async def cycle(self, s: AcquireTerms, run: StreamRun) -> tuple[Blocks | None, Stop | None]:
        # Acquisition.cycle waited out any earlier stream job of the session: no await before the permit.
        provider = self._providers[s.provider]
        try:
            rec = self._submit(s, run, provider)
        except StreamRefused:
            return None, Stop.REFUSED
        job = self._runner.start(rec, provider)
        try:
            done, _ = await asyncio.wait({job}, timeout=run.seconds_left())
        except asyncio.CancelledError:  # the stream was stopped: the job's bits go to the pool
            self._runner.discard(rec, job)  # also when it finished just before the stop landed
            raise
        if job not in done:  # the deadline came first
            self._runner.discard(rec, job)
            return None, Stop.ABANDONED
        try:
            m = job.result()
        except AppError:
            raise
        except Exception as exc:
            raise provider_failure(exc) from exc
        return [PoolBlock.of(m)], None

    def _submit(self, s: AcquireTerms, run: StreamRun, provider: Provider) -> JobRecord:
        # The permit: check, job record and count in one synchronous call, so a stalled loop cannot
        # slip a job past the deadline or the job limit. A ceiling (400 / 429) propagates uncounted.
        def submit() -> JobRecord:
            if self._runner.has_inflight(s.id, "stream"):  # the wait in cycle rules it out; a bug if it fires
                raise AppError(409, "stream_job_in_flight", IN_FLIGHT_DETAIL)
            return self._runner.submit(provider, s.n_qubits, s.shots_per_job, "stream", s.id)

        rec = run.submit(submit)
        self._bus.publish("stream", {"stream": s.stream_dict()}, s.id)
        return rec

    async def pace(self, s: AcquireTerms, run: StreamRun, t0: float) -> None:
        return None  # the next job starts when this result arrives: depth 1, no interval


class _BufferedSource:
    def __init__(self, pools: PoolStore, feeder: PoolFeeder, bus: EventBus) -> None:
        self._pools = pools
        self._feeder = feeder
        self._bus = bus

    def busy(self, s: AcquireTerms) -> bool:
        return False  # a pool take waits for no job

    async def settle(self, s: AcquireTerms, run: StreamRun) -> bool:
        return True

    async def ready(self, s: AcquireTerms) -> None:
        have = await asyncio.to_thread(self._pools.available, s.pool_key)
        if have < s.shots_per_job:
            await self._top_up(s, force=True)
            raise AppError(
                409, "pool_empty",
                f"pool {s.pool_key} has {have} unconsumed shots, stream needs {s.shots_per_job} per cycle",
            )

    async def one(self, s: AcquireTerms, kind: str) -> Blocks:
        return await self._take(s, 1, None)

    async def cycle(self, s: AcquireTerms, run: StreamRun) -> tuple[Blocks | None, Stop | None]:
        # Never given up: a take running at the deadline completes and is recorded (abandoning it
        # would drop pool shots); only its refill is gated by the run. A stop cannot drop them
        # either: the take (and the bookkeeping after it) runs on in its own task, the cycle waits
        # for it through any further cancel, and taken shots leave as CycleInterrupted.
        take = asyncio.ensure_future(self._take(s, s.shots_per_job, run))
        try:
            return await asyncio.shield(take), None
        except asyncio.CancelledError:
            while not take.done():
                try:
                    await asyncio.wait({take})  # never cancels the take, never raises its error
                except asyncio.CancelledError:
                    continue  # stopped again while the take finishes (file I/O; it will)
            if take.cancelled() or take.exception() is not None:
                raise  # the take itself failed: nothing was consumed (the store is unchanged)
            raise CycleInterrupted(blocks=take.result()) from None
        except AppError as exc:
            if exc.code == "pool_empty" and run.refusal:  # the refill that would have fed it was refused
                return None, Stop.REFUSED
            raise

    async def _take(self, s: AcquireTerms, shots: int, run: StreamRun | None) -> Blocks:
        try:
            blocks = await asyncio.to_thread(self._pools.take, s.pool_key, shots)  # file I/O off the loop
        except AppError as exc:
            if exc.code == "pool_empty":
                await self._top_up(s, force=True, run=run)  # the request still 409s; the refill starts
            raise
        # The shots are consumed: from here nothing may lose them. A failure of the bookkeeping
        # below is reported with its real text, and the shots are still returned.
        try:
            await self._top_up(s, run=run)
        except Exception as exc:
            self._report(s, "refill_check_failed", exc)
        try:
            pool = await asyncio.to_thread(self._pools.get, s.pool_key)
        except Exception as exc:
            self._report(s, "pool_read_failed", exc)
        else:
            if pool is not None:
                self._bus.publish("pool", {"pool": pool.summary()})
        return blocks

    def _report(self, s: AcquireTerms, code: str, exc: Exception) -> None:
        log.error("pool %s: %s after a take: %s", s.pool_key, code, exc)
        self._bus.publish("error", {"scope": "pool", "pool_key": s.pool_key, "error": code,
                                    "detail": f"{type(exc).__name__}: {exc}"}, s.id)

    async def _top_up(self, s: AcquireTerms, *, force: bool = False, run: StreamRun | None = None) -> None:
        """auto_refill for this session (PoolFeeder.top_up). A refill the stream caused shows in its jobs_used."""
        if await self._feeder.top_up(s, force=force, run=run) is TopUp.SUBMITTED and run is not None:
            self._bus.publish("stream", {"stream": s.stream_dict()}, s.id)

    async def pace(self, s: AcquireTerms, run: StreamRun, t0: float) -> None:
        elapsed = run.clock() - t0
        await asyncio.sleep(max(0.0, min(s.interval_ms / 1000 - elapsed, run.seconds_left())))


class Acquisition:
    """The one way a session gets bits: one per interaction (`one`) or one stream cycle (`cycle`)."""

    def __init__(self, providers: Mapping[str, Provider], runner: JobRunner, pools: PoolStore,
                 feeder: PoolFeeder, bus: EventBus) -> None:
        self._sources: dict[str, _Source] = {
            "live": _LiveSource(providers, runner, bus),
            "buffered": _BufferedSource(pools, feeder, bus),
        }

    async def one(self, s: AcquireTerms, kind: Literal["press", "predict", "perceive"]) -> Batch:
        """One shot for a key press, predict or perceive. Raises AppError (409 pool_empty, 400 / 429, 502)."""
        seg = s.segment
        blocks = await self._sources[seg.mode].one(s, kind)
        return _batch(s, seg, blocks)

    async def cycle(self, s: AcquireTerms, run: StreamRun) -> Cycle:
        """One stream cycle of shots_per_job shots. A limit is a Stop, never an exception.

        Live: when no stream job of the session is in flight, there is no await before the permit;
        otherwise the cycle first waits for that job (bounded by the deadline: Stop.EXPIRED), then
        captures the segment afresh. The run's limits are judged at the submit itself either way.
        """
        source = self._sources[s.segment.mode]
        if source.busy(s):
            # Depth 1, before the capture that decides the cycle, so a switch during the wait is obeyed.
            if not await source.settle(s, run):
                return Cycle(None, Stop.EXPIRED)
        seg = s.segment  # the capture; from here to the permit (live) there is no await
        source = self._sources[seg.mode]
        try:
            blocks, stop = await source.cycle(s, run)
        except CycleInterrupted as stopped:
            stopped.batch = _batch(s, seg, stopped.blocks)
            raise
        if blocks is None:
            return Cycle(None, stop)
        # The bits just acquired are real and are counted; a refill they needed may have been refused.
        stop = Stop.REFUSED if run.refusal else None

        async def pace(t0: float) -> None:
            await source.pace(s, run, t0)

        return Cycle(_batch(s, seg, blocks), stop, pace)

    async def admit_stream(self, s: AcquireTerms) -> None:
        """Judge whether a stream may start now. The caller creates the run with no await after this returns."""
        while True:
            mode = s.mode
            source = self._sources[mode]
            await source.ready(s)
            # From here to the return there is no await. The await above lets other requests run,
            # so everything is judged again against the state of now (a PATCH or a concurrent start).
            if s.streaming:
                raise AppError(409, "stream_running", "stream already running")
            if s.mode != mode:
                continue  # the mode changed while we awaited: the check above was for the other mode
            return


def _batch(s: AcquireTerms, seg: SegmentRef, blocks: Blocks) -> Batch:
    bits = np.concatenate([b.bits for b in blocks])
    first = blocks[0]
    provenance = {
        "job_id": first.job_id,
        "job_ids": list(dict.fromkeys(b.job_id for b in blocks)),
        "source": seg.mode,
        "provider": s.provider,
        "backend": first.backend,
        "measured_at": first.measured_at,
        "physical_qubits": list(first.physical_qubits),
        "pieces": [{"job_id": b.job_id, "backend": b.backend, "measured_at": b.measured_at,
                    "physical_qubits": list(b.physical_qubits), "shots": b.shots}
                   for b in blocks],
        "segment": seg.index,
        "shots": int(bits.shape[0]),
    }
    return Batch(bits, aggregate(bits, s.aggregation), provenance, seg)
