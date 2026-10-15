"""StreamRun on a fake clock: the permit at the submit, the count, and the snapshot the API returns.

No event loop and no sleeps: the clock is a number the test moves.
"""
import json

import pytest

from quantum_lite.errors import AppError
from quantum_lite.stream_run import StreamRefused, StreamRun


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def make(seconds=5, max_jobs=None, now=100.0):
    clock = Clock(now)
    return StreamRun(seconds, max_jobs, clock), clock


class Calls:
    """A submit fn that counts its calls and returns a fresh record each time."""

    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return f"rec-{self.n}"


# --- the deadline ----------------------------------------------------------------------------
def test_the_deadline_reached_exactly_refuses_and_does_not_submit():
    run, clock = make(seconds=5, now=100.0)
    fn = Calls()
    clock.now = 105.0  # deadline = construction time + seconds, to the tick
    with pytest.raises(StreamRefused) as err:
        run.submit(fn)
    assert (err.value.code, err.value.detail) == ("limit_seconds", "the 5 s stream duration is over")
    assert str(err.value) == "limit_seconds: the 5 s stream duration is over"
    assert fn.n == 0 and run.jobs_used == 0
    assert run.refusal == ("limit_seconds", "the 5 s stream duration is over")
    assert run.expired()


def test_one_tick_before_the_deadline_submits_and_counts():
    run, clock = make(seconds=5, now=100.0)
    fn = Calls()
    clock.now = 104.999
    assert not run.expired()
    assert run.submit(fn) == "rec-1"  # fn's result comes back
    assert fn.n == 1 and run.jobs_used == 1 and run.refusal is None


def test_the_deadline_is_fixed_when_the_run_is_made():
    run, clock = make(seconds=5, now=100.0)
    clock.now = 103.0
    run.submit(Calls())  # a submit does not move the deadline
    clock.now = 105.0
    with pytest.raises(StreamRefused):
        run.submit(Calls())


# --- the job limit ---------------------------------------------------------------------------
def test_the_job_limit_reached_exactly_refuses_and_does_not_submit():
    run, _ = make(max_jobs=2)
    fn = Calls()
    run.submit(fn)
    run.submit(fn)
    with pytest.raises(StreamRefused) as err:
        run.submit(fn)
    assert (err.value.code, err.value.detail) == ("limit_jobs", "2 jobs caused by this stream, the limit is 2")
    assert fn.n == 2 and run.jobs_used == 2


def test_refused_submits_are_never_counted():
    run, clock = make(seconds=5, max_jobs=1)
    fn = Calls()
    run.submit(fn)
    for _ in range(3):
        with pytest.raises(StreamRefused):
            run.submit(fn)
    clock.now = 200.0
    with pytest.raises(StreamRefused):
        run.submit(fn)
    assert fn.n == 1 and run.jobs_used == 1 and run.snapshot(running=True)["jobs_used"] == 1


def test_a_submit_that_raises_is_not_counted_and_propagates():
    run, _ = make(max_jobs=1)

    def ceiling():
        raise AppError(409, "ibm_max_jobs", "IBM_MAX_JOBS reached")

    with pytest.raises(AppError) as err:
        run.submit(ceiling)
    assert err.value.code == "ibm_max_jobs"  # not turned into a refusal
    assert run.jobs_used == 0 and run.refusal is None
    assert run.submit(Calls()) == "rec-1"  # the one permitted job is still there
    assert run.jobs_used == 1


def test_no_job_limit_never_refuses_on_count():
    run, _ = make(seconds=5, max_jobs=None)
    fn = Calls()
    for _ in range(1000):
        run.submit(fn)
    assert fn.n == 1000 and run.jobs_used == 1000 and run.refusal is None


# --- the pending refusal and how the run ends ---------------------------------------------------
def test_the_last_refusal_wins():
    run, clock = make(seconds=5, max_jobs=1)
    run.submit(Calls())
    with pytest.raises(StreamRefused):
        run.submit(Calls())
    assert run.refusal[0] == "limit_jobs"
    clock.now = 105.0
    with pytest.raises(StreamRefused):
        run.submit(Calls())
    assert run.refusal == ("limit_seconds", "the 5 s stream duration is over")
    assert run.end_by_refusal() == ("limit_seconds", "the 5 s stream duration is over")
    assert run.snapshot(running=False)["stopped_reason"] == "limit_seconds: the 5 s stream duration is over"


def test_end_by_refusal_without_a_refusal_is_a_bug_not_a_stop():
    run, _ = make()
    with pytest.raises(RuntimeError):
        run.end_by_refusal()
    assert run.snapshot(running=False)["stopped_reason"] is None


def test_seconds_left_clamps_at_zero_and_is_shown_only_while_running():
    run, clock = make(seconds=5, now=100.0)
    clock.now = 101.23456
    assert run.seconds_left() == pytest.approx(3.76544)
    assert run.snapshot(running=True)["seconds_left"] == 3.765  # rounded to ms, as the API has it
    assert run.snapshot(running=False)["seconds_left"] is None
    clock.now = 110.0
    assert run.seconds_left() == 0.0
    assert run.snapshot(running=True)["seconds_left"] == 0.0


# --- the snapshot: the session's "stream" dict, byte for byte ------------------------------------
def dumps(snapshot):
    return json.dumps(snapshot)  # key order included: the API returns the dict as it is built


def test_snapshot_of_a_fresh_running_run():
    run, clock = make(seconds=30, max_jobs=5, now=100.0)
    clock.now = 100.5
    assert dumps(run.snapshot(running=True)) == (
        '{"running": true, "error": null, "stopped_reason": null, "seconds_limit": 30, '
        '"seconds_left": 29.5, "jobs_limit": 5, "jobs_used": 0}')


def test_snapshot_after_a_deadline_stop():
    run, clock = make(seconds=5, max_jobs=None)
    run.submit(Calls())
    clock.now = 106.0
    assert run.end_by_deadline() == ("limit_seconds", "the 5 s stream duration is over")
    assert dumps(run.snapshot(running=False)) == (
        '{"running": false, "error": null, "stopped_reason": "limit_seconds: the 5 s stream duration is over", '
        '"seconds_limit": 5, "seconds_left": null, "jobs_limit": null, "jobs_used": 1}')


def test_snapshot_after_a_deadline_stop_with_a_job_in_flight():
    run, _ = make(seconds=1, max_jobs=3)
    code, detail = run.end_by_deadline(job_in_flight=True)
    assert code == "limit_seconds"
    assert detail == ("the 1 s stream duration is over; "
                      "the job in flight cannot be recalled, it finishes and its bits go to the pool")
    assert run.snapshot(running=False)["stopped_reason"] == f"limit_seconds: {detail}"


def test_snapshot_after_a_job_limit_stop():
    run, _ = make(seconds=60, max_jobs=2)
    run.submit(Calls())
    run.submit(Calls())
    with pytest.raises(StreamRefused):
        run.submit(Calls())
    assert run.end_by_refusal() == ("limit_jobs", "2 jobs caused by this stream, the limit is 2")
    assert dumps(run.snapshot(running=False)) == (
        '{"running": false, "error": null, "stopped_reason": "limit_jobs: 2 jobs caused by this stream, '
        'the limit is 2", "seconds_limit": 60, "seconds_left": null, "jobs_limit": 2, "jobs_used": 2}')


def test_snapshot_after_an_error():
    run, _ = make(seconds=30)
    run.end_by_error("pool_empty: pool x has 5 unconsumed shots, 20 requested")
    assert dumps(run.snapshot(running=False)) == (
        '{"running": false, "error": "pool_empty: pool x has 5 unconsumed shots, 20 requested", '
        '"stopped_reason": null, "seconds_limit": 30, "seconds_left": null, "jobs_limit": null, "jobs_used": 0}')
