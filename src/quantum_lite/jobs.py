"""Job registry and runner: every job this process submits is recorded and published."""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

from quantum_lite.budget import IbmBudget
from quantum_lite.errors import AppError, ProviderError
from quantum_lite.events import EventBus, now_iso
from quantum_lite.providers.base import JOB_STATES, Measurement, Provider

log = logging.getLogger("quantum_lite.jobs")

# Estimate only: 2 s fixed + 0.00035 s per shot (docs/DESIGN.md). Not a measured number.
ESTIMATE_BASE_SECONDS = 2
ESTIMATE_SECONDS_PER_SHOT = 0.00035
ESTIMATE_FORMULA = "2 + 0.00035 * shots"

# Shot cap per job for every session's shots_per_job and every non-ibm prefetch. An ibm
# prefetch is bounded by IBM_MAX_SHOTS_PER_JOB instead.
MAX_SHOTS_PER_JOB = 1_000_000

# /api/jobs lists every ibm job (each one is QPU time the user caused). Finished sim jobs are
# free and a live sim stream makes dozens per second, so only the newest are kept.
MAX_FINISHED_SIM_RECORDS = 500


def estimate_qpu_seconds(shots: int) -> float:
    return ESTIMATE_BASE_SECONDS + ESTIMATE_SECONDS_PER_SHOT * shots


@dataclass
class JobRecord:
    id: str
    provider: str
    kind: str  # stream | press | predict | perceive | prefetch
    shots: int
    n_qubits: int
    session_id: str | None
    submitted_at: str = field(default_factory=now_iso)
    state: str = "submitting"  # local: transpile/submit in progress; the provider sets "queued"
    job_id: str | None = None  # the provider's id, known once submitted
    backend: str | None = None
    queue_position: int | None = None
    finished_at: str | None = None
    error: str | None = None
    simulated_latency: bool = False
    discarded: bool = False  # the caller went away; the job still ran
    pooled_to: str | None = None  # key of the pool that kept a discarded job's bits

    def to_dict(self) -> dict:
        is_qpu = self.provider == "ibm"
        return {
            "id": self.id,
            "job_id": self.job_id,
            "provider": self.provider,
            "backend": self.backend,
            "kind": self.kind,
            "shots": self.shots,
            "n_qubits": self.n_qubits,
            "session_id": self.session_id,
            "state": self.state,
            "queue_position": self.queue_position,
            "submitted_at": self.submitted_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "simulated_latency": self.simulated_latency,
            "discarded": self.discarded,
            "pooled_to": self.pooled_to,
            "estimated_qpu_seconds": estimate_qpu_seconds(self.shots) if is_qpu else None,
            "estimate_formula": ESTIMATE_FORMULA if is_qpu else None,
        }


class JobRunner:
    def __init__(self, bus: EventBus, budget: IbmBudget, sim_latency_seconds: float) -> None:
        self.bus = bus
        self.budget = budget
        self._sim_latency = sim_latency_seconds
        self.records: dict[str, JobRecord] = {}
        self.stop = threading.Event()  # set at shutdown so IBM polling threads exit
        self._tasks: dict[str, asyncio.Task] = {}  # record id -> the task running the job
        self.max_finished_sim_records = MAX_FINISHED_SIM_RECORDS
        # Called with (record, measurement) when a job finishes after its caller was cancelled
        # (stream stopped mid-job): the bits were measured and must not vanish. PoolFeeder installs it.
        self.on_discarded: Callable[[JobRecord, Measurement], None] | None = None

    def list(self) -> list[dict]:
        return [r.to_dict() for r in self.records.values()]

    def has_inflight(self, session_id: str, kind: str) -> bool:
        """True while a job of this kind for the session has not finished (even if its caller left)."""
        return any(r.session_id == session_id and r.kind == kind and r.finished_at is None
                   for r in self.records.values())

    async def settled(self, session_id: str, kind: str, timeout: float) -> bool:
        """Wait at most `timeout` s for the started jobs of this kind for the session to finish.

        True when none is in flight any more. Waiting does not cancel or discard anything.
        """
        tasks = {self._tasks[r.id] for r in self.records.values()
                 if r.session_id == session_id and r.kind == kind and r.finished_at is None and r.id in self._tasks}
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        return not self.has_inflight(session_id, kind)

    def _publish(self, rec: JobRecord) -> None:
        self.bus.publish("job", rec.to_dict(), session_id=rec.session_id)

    def submit(self, provider: Provider, n: int, shots: int, kind: str, session_id: str | None) -> JobRecord:
        """Check ceilings (IBM only), reserve budget and record the job. Raises AppError."""
        if provider.id == "ibm":
            self.budget.reserve(shots)
        rec = JobRecord(
            id=uuid.uuid4().hex[:12], provider=provider.id, kind=kind, shots=shots,
            n_qubits=n, session_id=session_id,
            simulated_latency=provider.id == "sim" and self._sim_latency > 0,
        )
        self.records[rec.id] = rec
        self._publish(rec)
        return rec

    async def execute(self, rec: JobRecord, provider: Provider) -> Measurement:
        """Run the job and return its measurement. Cancelling the caller does not stop the job: see discard."""
        task = self.start(rec, provider)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.discard(rec, task)
            raise

    def start(self, rec: JobRecord, provider: Provider) -> asyncio.Task:
        """Start the job now and return its task. Synchronous: the job runs whatever happens to the caller.

        A caller that stops waiting for it calls discard.
        """
        loop = asyncio.get_running_loop()

        def on_state(state: str, *, queue_position=None, job_id=None, backend=None) -> None:
            if state not in JOB_STATES:
                raise ProviderError(f"provider reported unknown job state {state!r}")
            loop.call_soon_threadsafe(self._apply, rec, state, queue_position, job_id, backend)

        async def work() -> Measurement:
            try:
                meas = await asyncio.to_thread(provider.run, rec.n_qubits, rec.shots, on_state, self.stop)
            except Exception as exc:
                rec.state = "error"
                rec.error = f"{type(exc).__name__}: {exc}"
                rec.finished_at = now_iso()
                self._publish(rec)
                log.error("job %s failed: %s", rec.id, rec.error)
                raise
            rec.state = "done"
            rec.job_id = meas.job_id
            rec.backend = meas.backend
            rec.queue_position = None
            rec.finished_at = now_iso()
            self._publish(rec)
            self._prune()
            return meas

        task = asyncio.ensure_future(work())
        self._tasks[rec.id] = task

        def _done(t: asyncio.Task) -> None:
            self._tasks.pop(rec.id, None)
            if not t.cancelled() and t.exception() is None and rec.discarded and self.on_discarded is not None:
                self.on_discarded(rec, t.result())
            # (t.exception() above also marks a failure retrieved; it is in the record and the log)

        task.add_done_callback(_done)
        return task

    def discard(self, rec: JobRecord, task: asyncio.Task | None = None) -> None:
        """The caller went away: the job still runs, and when it finishes on_discarded gets its bits.

        `task` is the job's task (from start). When it already finished and its done-callback has run
        (the caller was cancelled after the job ended, before it took the result), the bits are handed
        to on_discarded here, once: they were measured and must not vanish.
        """
        if rec.discarded:
            return  # already handed over (or to be): at most one adoption
        rec.discarded = True
        if task is None or not task.done() or rec.id in self._tasks:
            return  # still running, or its done-callback is yet to run and will see `discarded`
        if not task.cancelled() and task.exception() is None and self.on_discarded is not None:
            self.on_discarded(rec, task.result())

    def mark_pooled(self, rec: JobRecord, key: str) -> None:
        """A discarded job's bits were kept in pool `key`: record it and say so."""
        rec.pooled_to = key
        self._publish(rec)

    def _prune(self) -> None:
        finished = [r.id for r in self.records.values() if r.provider != "ibm" and r.finished_at is not None]
        for rid in finished[: max(0, len(finished) - self.max_finished_sim_records)]:
            del self.records[rid]

    def _apply(self, rec: JobRecord, state: str, queue_position, job_id, backend) -> None:
        if rec.state in ("done", "error"):
            return  # a late progress callback must not reopen a finished job
        rec.state = state
        rec.queue_position = queue_position
        if job_id:
            rec.job_id = job_id
        if backend:
            rec.backend = backend
        self._publish(rec)

    async def run(self, provider: Provider, n: int, shots: int, kind: str, session_id: str | None) -> Measurement:
        rec = self.submit(provider, n, shots, kind, session_id)
        return await self.execute(rec, provider)
