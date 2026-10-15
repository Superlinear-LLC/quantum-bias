"""One stream run: its limits, the jobs it caused, and how it ended.

This is where docs/DESIGN.md "Cost controls" is enforced: "the check sits at the submit itself".
Every job a stream run causes (a live cycle job, or on sim an auto_refill job) is submitted
through StreamRun.submit, which checks the deadline and the job limit, submits and counts in one
synchronous step. With no await between them, a stalled event loop cannot slip a job past either.

The run publishes no events; the session manager does that from what the run returns.
"""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

IN_FLIGHT = "; the job in flight cannot be recalled, it finishes and its bits go to the pool"


class StreamRefused(Exception):
    """A stream limit refused a job at the submit boundary. Not a fault: the run stops by it."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class StreamRun:
    """One start-to-stop of a stream, bounded by `seconds` and, when set, `max_jobs`.

    `clock` returns monotonic seconds (time.monotonic in the app). The deadline is fixed at
    construction: nothing the run causes is submitted at or after it, nor beyond max_jobs.
    jobs_used counts every job the run caused: its live cycle jobs plus the auto_refill jobs it
    triggered. Not the refills that press / predict / perceive trigger, and not a refill that was
    already running when the run began.
    """

    def __init__(self, seconds: int, max_jobs: int | None, clock: Callable[[], float]) -> None:
        self.seconds = seconds
        self.max_jobs = max_jobs
        self.clock = clock
        self.deadline = clock() + seconds
        self.jobs_used = 0
        self._refusal: tuple[str, str] | None = None  # the last submit a limit refused, pending until the run ends
        self._stopped: str | None = None  # "limit_seconds: ..." / "limit_jobs: ...": a limit ended the run
        self._error: str | None = None

    # --- the permit ------------------------------------------------------------------
    def submit(self, fn: Callable[[], T]) -> T:
        """Check the limits, call fn (the submit) and count it. Synchronous on purpose.

        A refusal is recorded as pending and raised as StreamRefused; fn is not called. If fn
        raises (a budget ceiling, say), nothing is counted and the exception propagates.
        """
        if self.clock() >= self.deadline:
            self._refuse("limit_seconds", f"the {self.seconds} s stream duration is over")
        if self.max_jobs is not None and self.jobs_used >= self.max_jobs:
            self._refuse("limit_jobs", f"{self.jobs_used} jobs caused by this stream, the limit is {self.max_jobs}")
        result = fn()
        self.jobs_used += 1
        return result

    def _refuse(self, code: str, detail: str):
        self._refusal = (code, detail)  # last one wins
        raise StreamRefused(code, detail)

    @property
    def refusal(self) -> tuple[str, str] | None:
        return self._refusal

    def expired(self) -> bool:
        return self.clock() >= self.deadline

    def seconds_left(self) -> float:
        return max(0.0, self.deadline - self.clock())

    # --- how the run ended -------------------------------------------------------------
    def end_by_deadline(self, job_in_flight: bool = False) -> tuple[str, str]:
        detail = f"the {self.seconds} s stream duration is over" + (IN_FLIGHT if job_in_flight else "")
        return self._end("limit_seconds", detail)

    def end_by_refusal(self) -> tuple[str, str]:
        if self._refusal is None:
            raise RuntimeError("end_by_refusal without a refused submit")
        return self._end(*self._refusal)

    def end_by_error(self, message: str) -> None:
        self._error = message

    def _end(self, code: str, detail: str) -> tuple[str, str]:
        self._stopped = f"{code}: {detail}"
        return code, detail

    def snapshot(self, running: bool) -> dict:
        """The session's "stream" dict for this run."""
        return {
            "running": running, "error": self._error, "stopped_reason": self._stopped,
            "seconds_limit": self.seconds, "seconds_left": round(self.seconds_left(), 3) if running else None,
            "jobs_limit": self.max_jobs, "jobs_used": self.jobs_used,
        }
