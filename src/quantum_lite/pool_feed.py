"""Pool feeding: every job that fills a pool, and the one way its bits get into one.

Three ways bits reach a pool, all through `bank`:
- `PoolFeeder.prefetch`: an explicit prefetch (POST /api/pools/prefetch), a job the user caused,
  listed in /api/jobs. With the CLI's `prefetch`, the only way an ibm pool is refilled
  (auto_refill never runs on ibm).
- `PoolFeeder.top_up`: auto_refill on behalf of one session (sim only). A refill a stream causes
  is submitted through StreamRun.submit, so the stream's limits check and count it. A failed
  refill is never retried: it switches auto_refill off for that session.
- the discarded-job sink: a live job whose caller went away still ran; its bits go to the pool
  of the device that measured them.

The CLI shares the sync core (lookup_provider, check_prefetch, bank), not the feeder: it has no
event loop, no runner and no bus, and keeps its own step order and prints.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Mapping, Protocol

from quantum_lite.budget import IbmBudget
from quantum_lite.circuit import validate_qubits
from quantum_lite.errors import AppError, off_loop
from quantum_lite.events import EventBus
from quantum_lite.jobs import MAX_SHOTS_PER_JOB, JobRecord, JobRunner
from quantum_lite.pools import Pool, PoolBlock, PoolStore, pool_key
from quantum_lite.providers.base import Measurement, Provider
from quantum_lite.stream_run import StreamRefused, StreamRun

log = logging.getLogger("quantum_lite.pool_feed")

NO_RETRY = " (auto_refill switched off for this session; no automatic retry)"


# --- the sync core (server and CLI) ------------------------------------------------------
def lookup_provider(providers: Mapping[str, Provider], provider_id: str) -> Provider:
    provider = providers.get(provider_id)
    if provider is None:
        raise AppError(400, "unknown_provider", f"unknown provider {provider_id!r}; known: {sorted(providers)}")
    info = provider.info()
    if not info["available"]:
        raise AppError(409, "provider_unavailable", f"provider {provider_id} unavailable: {info['reason']}")
    return provider


def check_prefetch(providers: Mapping[str, Provider], provider_id: str, n: int, shots: int,
                   budget: IbmBudget) -> Provider:
    """Refuse a prefetch that cannot run, before anything is resolved or recorded. Counts nothing."""
    provider = lookup_provider(providers, provider_id)
    validate_qubits(n)
    if shots < 1:
        raise AppError(400, "invalid_shots", f"shots must be >= 1, got {shots}")
    if provider.id == "ibm":
        budget.check_job_size(shots)
    elif shots > MAX_SHOTS_PER_JOB:
        raise AppError(400, "invalid_shots", f"shots must be 1..{MAX_SHOTS_PER_JOB}, got {shots}")
    return provider


def bank(store: PoolStore, provider_id: str, n: int, m: Measurement, *, expect: str | None) -> Pool:
    """Append a measurement to its pool. Blocking (takes the store's file lock): call off the loop.

    `expect`: the implementation the pool is for; a job that ran elsewhere is refused and the
    store is unchanged. None (bits salvaged from a discarded job): the pool is the device that
    measured them.
    """
    if expect is not None and m.backend != expect:
        raise ValueError(f"job ran on {m.backend}, pool is for {expect}")
    return store.append(provider_id, m.backend, n, PoolBlock.of(m))


# --- the server's feeder -------------------------------------------------------------------
class TopUp(Enum):
    IDLE = "idle"  # auto_refill off, or the pool is not low
    BUSY = "busy"  # a refill of this pool is already in flight
    SUBMITTED = "submitted"
    REFUSED_BY_RUN = "refused_by_run"  # a stream limit refused it; the refusal is pending on the run
    SWITCHED_OFF = "switched_off"  # the submit was refused (a ceiling); auto_refill is now off


class RefillTerms(Protocol):
    """What a refill needs from a session. Session satisfies it; this module never imports sessions."""

    id: str
    provider: str
    implementation: str
    n_qubits: int
    pool_key: str
    refill_below: int
    refill_shots: int
    auto_refill: bool  # the feeder may set it False, never True


@dataclass(frozen=True)
class Placed:
    record: JobRecord
    pool_key: str


class PoolFeeder:
    def __init__(self, providers: Mapping[str, Provider], store: PoolStore, runner: JobRunner,
                 bus: EventBus, budget: IbmBudget) -> None:
        self.providers = providers
        self.store = store
        self.runner = runner
        self.bus = bus
        self.budget = budget
        # pool key -> id of the refill job that holds it. Only that job's landing releases it,
        # so a manual prefetch of the same pool cannot clear it.
        self._inflight: dict[str, str] = {}
        self._landings: set[asyncio.Task] = set()
        runner.on_discarded = self._adopt

    async def prefetch(self, provider_id: str, n: int, shots: int) -> Placed:
        """Submit one prefetch job; its bits land in the background. Raises before anything is recorded."""
        provider = check_prefetch(self.providers, provider_id, n, shots, self.budget)
        implementation = await off_loop(provider.resolve)
        key = pool_key(provider.id, implementation, n)
        # No await from here to the return: the record still reads "submitting" when the caller sees it.
        if self.runner.stop.is_set():
            # Shutdown began while resolve ran: nothing would wait for or bank the job (on ibm, billed).
            raise AppError(503, "shutting_down", "the server is shutting down; the prefetch was not submitted")
        rec = self.runner.submit(provider, n, shots, "prefetch", None)  # an ibm ceiling raises 429 here
        self._spawn(self._land(rec, provider, implementation, key, None))
        return Placed(rec, key)

    async def top_up(self, terms: RefillTerms, *, force: bool = False, run: StreamRun | None = None) -> TopUp:
        """Start one refill job when auto_refill is on and the pool is low.

        `force`: a request just found the pool short, whatever refill_below says.
        `run`: the refill is caused by a stream. Its submit goes through run.submit, which checks the
        run's deadline and job limit and counts it. A limit that refuses it leaves the refusal pending
        on the run and submits nothing; auto_refill stays as the session had it (a limit is the user's
        choice, not a fault of the refill) and the stream stops by that limit. A refill already
        running (the guard) is neither started nor counted again.
        """
        key = terms.pool_key
        if not terms.auto_refill:
            return TopUp.IDLE
        if key in self._inflight:
            return TopUp.BUSY
        if not force and await asyncio.to_thread(self.store.available, key) >= terms.refill_below:
            return TopUp.IDLE
        # From here to the spawn there is no await. Judge both again: while we awaited, another
        # request may have started a refill, or one may have failed and switched auto_refill off
        # (a failed refill is never retried).
        if not terms.auto_refill:
            return TopUp.IDLE
        if key in self._inflight:
            return TopUp.BUSY
        if terms.provider == "ibm":
            # create refuses it (400 auto_refill_ibm): every ibm job is an explicit, user-caused prefetch.
            raise ValueError(f"auto_refill on ibm pool {key}: ibm pools are filled only by an explicit prefetch")
        provider = self.providers[terms.provider]

        def submit() -> JobRecord:
            return self.runner.submit(provider, terms.n_qubits, terms.refill_shots, "prefetch", terms.id)

        try:
            rec = submit() if run is None else run.submit(submit)
        except StreamRefused:
            return TopUp.REFUSED_BY_RUN
        except AppError as exc:
            # Refused before submission (a ceiling): it would be refused again on every take,
            # so switch the refill off and say so; the pool stays as it is.
            terms.auto_refill = False
            self.bus.publish("error", {
                "scope": "refill", "pool_key": key, "error": exc.code, "detail": f"{exc.detail}{NO_RETRY}",
            }, terms.id)
            return TopUp.SWITCHED_OFF
        self._inflight[key] = rec.id
        self._spawn(self._land(rec, provider, terms.implementation, key, terms))
        return TopUp.SUBMITTED

    async def drain(self) -> None:
        """Wait for every landing in flight (and any it leads to) to finish."""
        while self._landings:
            await asyncio.wait(set(self._landings))

    def close(self) -> Awaitable[list]:
        """Cancel every landing now; the returned gather completes when they have stopped."""
        tasks = list(self._landings)
        for t in tasks:
            t.cancel()
        return asyncio.gather(*tasks, return_exceptions=True)

    # --- background --------------------------------------------------------------------
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._landings.add(task)
        task.add_done_callback(self._landings.discard)

    async def _land(self, rec: JobRecord, provider: Provider, implementation: str, key: str,
                    terms: RefillTerms | None) -> None:
        """Run one prefetch or refill job and bank its bits. Never retried."""
        try:
            m = await self.runner.execute(rec, provider)
            pool = await asyncio.to_thread(bank, self.store, provider.id, rec.n_qubits, m, expect=implementation)
            self.bus.publish("pool", {"pool": pool.summary(), "job_id": rec.id})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, AppError) else "prefetch_error"
            log.error("prefetch %s failed: %s", rec.id, exc)
            detail = f"{type(exc).__name__}: {exc}"
            if terms is not None:
                # Never retry a failed refill by itself (on ibm every retry is a QPU job).
                terms.auto_refill = False
                detail += NO_RETRY
            self.bus.publish("error", {"scope": "prefetch", "pool_key": key, "job_id": rec.id,
                                       "error": code, "detail": detail}, terms.id if terms else None)
        finally:
            if self._inflight.get(key) == rec.id:  # only the refill that took the guard releases it
                del self._inflight[key]

    def _adopt(self, rec: JobRecord, m: Measurement) -> None:
        """A live job finished after its caller was cancelled (stream stopped mid-job).

        Its bits are real measurements that were paid for: they go to the pool of the device that
        measured them, so they can be consumed later as buffered bits. They are not counted in
        any live stats, and nothing is invented. A prefetch job banks its own bits, and nothing is
        pooled once shutdown has begun.
        """
        if rec.kind == "prefetch" or self.runner.stop.is_set():
            return
        self._spawn(self._salvage(rec, m))

    async def _salvage(self, rec: JobRecord, m: Measurement) -> None:
        try:
            pool = await asyncio.to_thread(bank, self.store, m.provider, rec.n_qubits, m, expect=None)
        except Exception as exc:  # reported to the UI and the log, with the real text
            log.error("could not pool the bits of discarded job %s: %s", rec.id, exc)
            self.bus.publish("error", {"scope": "discarded_job", "job_id": rec.id, "error": "pool_write_failed",
                                       "detail": f"{type(exc).__name__}: {exc}"}, rec.session_id)
            return
        self.runner.mark_pooled(rec, pool.key)
        self.bus.publish("pool", {"pool": pool.summary(), "job_id": rec.id})
