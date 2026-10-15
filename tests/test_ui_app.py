"""app.js against a real server: what the focus view renders after a key press, a stream batch,
a guess and a reveal.

tests/ui_harness.js loads the real logic.js and app.js over a small DOM built from the real
index.html, talks real HTTP and real SSE to a real uvicorn server (cli.make_server), and prints
what app.js rendered. The only scripted part is where the bits come from: the sim provider's jobs
measure known bits (fakes.ScriptedSim), a buffered session takes them from a seeded pool, and ibm
runs over the FakeIbm boundary. So the expected display can be computed here, independently of
app.js. Needs node; skips without.
"""
import json
import math
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest

from conftest import ibm_app, make_settings, wait_for
from fakes import FakeIbm, ScriptedSim
from quantum_lite import cli
from quantum_lite.app import build_providers, create_app
from quantum_lite.errors import ProviderError
from quantum_lite.pools import PoolBlock
from quantum_lite.stats import bias_pct, z_score

HARNESS = Path(__file__).resolve().parent / "ui_harness.js"
THRESHOLD = 5.0  # the focus view's track half-width, in percent


@pytest.fixture
def node():
    path = shutil.which("node")
    if path is None:
        pytest.skip("node not installed; the app.js tests need it (the app itself does not)")
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


STREAM_DELAY = 0.4  # seconds per scripted live stream job, so the harness can look at each batch
SLOW_IBM = ("QUEUED",) * 40 + ("RUNNING", "DONE")  # about 0.4 s per FakeIbm job at poll_seconds=0.01


def scripted_app(tmp_path, bits=(), source="live", **sim_kw):
    """An app whose sim jobs measure `bits` (one entry per job), or, buffered, whose pool holds them.

    The UI asks for one qubit (aggregation none), so the pool is sim__aer_ideal__1.
    """
    settings = make_settings(tmp_path)
    if source == "buffered":
        app = create_app(settings)
        for k, b in enumerate(bits):
            app.state.mgr.pools.append("sim", "aer_ideal", 1, PoolBlock(
                f"seeded-{k}", "aer_ideal", (0,), "2026-01-01T00:00:00+00:00", np.array(b, dtype=np.uint8).reshape(-1, 1)))
        return app
    return create_app(settings, providers={**build_providers(settings), "sim": ScriptedSim(settings, bits, **sim_kw)})


class Server:
    def __init__(self, app):
        self.port = free_port()
        self.server = cli.make_server(app, "127.0.0.1", self.port)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        wait_for(lambda: self.server.started, what="server start")

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)
        assert not self.thread.is_alive(), "server did not stop"


def drive(node, tmp_path, scenario, params, bits=(), **sim_kw):
    """Run one harness scenario; return what app.js rendered (plus the server's job list) and the session."""
    if params.get("provider") == "ibm":  # the real IbmProvider over the fake IBM boundary, Aer bits
        app = ibm_app(tmp_path, FakeIbm(statuses=SLOW_IBM))
    else:
        app = scripted_app(tmp_path, bits, params.get("source", "live"), **sim_kw)
    srv = Server(app)
    try:
        r = subprocess.run([node, str(HARNESS), srv.base, scenario, json.dumps(params)],
                           capture_output=True, text=True, timeout=90)
        assert r.returncode == 0, f"harness failed:\n{r.stderr}"
        out = json.loads(r.stdout)
        server_session = httpx.get(f"{srv.base}/api/sessions/{out['sid']}").json()
        out["jobs"] = httpx.get(f"{srv.base}/api/jobs").json()
    finally:
        srv.stop()
    return out, server_session


def drive_stream(node, tmp_path, scenario, params, batches):
    """A live stream over the scripted batches; with them used up the next job waits (the test stops it)."""
    return drive(node, tmp_path, scenario, params, batches, delay=STREAM_DELAY, when_empty="block")


# ---------------------------------------------------------------- expected display, computed here
def signed_pct(x, digits):
    return f"{'+' if x >= 0 else ''}{x:.{digits}f}%"


def streak_text(n):
    if n < 5:
        return ""
    label = "LEGENDARY" if n >= 20 else "Mythic" if n >= 15 else "Epic" if n >= 10 else "Lucky"
    return f"{n} in a row — {label}"


def ball_target(bias):
    clamped = max(-THRESHOLD, min(THRESHOLD, bias))
    return (clamped + THRESHOLD) / (2 * THRESHOLD) * 100


def trailing_run(bits, target):
    n = 0
    for b in reversed(bits):
        if b != target:
            break
        n += 1
    return n


def assert_bias_view(snap, bits, streak, target):
    n, ones = len(bits), sum(bits)
    bias, z = bias_pct(ones, n), z_score(ones, n)
    assert snap["shots"] == n
    assert snap["big"] == signed_pct(bias, 3), snap["label"]
    assert snap["sigma"] == f"{z:.2f}σ", snap["label"]
    assert snap["left"] == (signed_pct(bias, 2) if bias < 0 else ""), snap["label"]
    assert snap["right"] == (signed_pct(bias, 2) if bias > 0 else ""), snap["label"]
    assert snap["ballTarget"] == pytest.approx(ball_target(bias)), snap["label"]
    want = streak_text(streak)
    if want:
        assert want in snap["streak"], (snap["label"], snap["streak"])
    else:
        assert snap["streak"] == "", (snap["label"], snap["streak"])
    assert snap["title"] == f"Target: {target}"
    assert snap["alerts"] == [], snap["label"]


# ---------------------------------------------------------------- tests
def test_key_press_renders_bias_sigma_ball_and_streak_from_each_response(node, tmp_path):
    bits = [0, 1, 1, 1, 1, 1, 0] + [1] * 10 + [0]  # a run of 5, a break, a run of 10, a break
    out, server = drive(node, tmp_path, "press", {"presses": len(bits), "target": 1}, [[b] for b in bits])
    assert len(out["snaps"]) == len(bits)
    for k, snap in enumerate(out["snaps"], start=1):
        assert_bias_view(snap, bits[:k], trailing_run(bits[:k], 1), target=1)
    streaks = [trailing_run(bits[:k], 1) for k in range(1, len(bits) + 1)]
    assert 5 in streaks and 10 in streaks and streaks[-1] == 0  # the script does cross both tiers
    last = out["snaps"][-1]
    assert last["big"] == signed_pct(server["stats"]["total"]["aggregated"]["bias_pct"], 3)
    assert last["sigma"] == f"{server['stats']['total']['aggregated']['z']:.2f}σ"
    last_job = [j for j in out["jobs"] if j["kind"] == "press"][-1]
    assert last["provenance"].startswith("sim/aer_ideal · live · ") and f"job {last_job['job_id']}" in last["provenance"]


def test_key_press_with_target_zero_counts_zeros(node, tmp_path):
    bits = [1, 0, 0, 0, 0, 0]
    out, _ = drive(node, tmp_path, "press", {"presses": len(bits), "target": 0}, [[b] for b in bits])
    for k, snap in enumerate(out["snaps"], start=1):
        assert_bias_view(snap, bits[:k], trailing_run(bits[:k], 0), target=0)
    assert "5 in a row" in out["snaps"][-1]["streak"]


def test_stream_batches_continue_and_reset_the_streak(node, tmp_path):
    batches = [[0] * 17 + [1] * 3, [1] * 20, [1] * 19 + [0]]
    out, server = drive_stream(node, tmp_path, "stream", {"total": 60, "target": 1}, batches)
    by_shots = {s["shots"]: s for s in out["snaps"] if s["label"].startswith("shots")}
    assert sorted(by_shots) == [20, 40, 60]
    flat = [b for batch in batches for b in batch]
    assert_bias_view(by_shots[20], flat[:20], 3, target=1)  # a tail of 3: below the message threshold
    assert_bias_view(by_shots[40], flat[:40], 23, target=1)  # whole batch is the target: continues
    assert_bias_view(by_shots[60], flat[:60], 0, target=1)  # ends on a miss
    assert "stream stopped" in out["snaps"][-1]["provenance"]
    assert server["stream"]["running"] is False


def test_predict_renders_each_result_and_the_hit_streak(node, tmp_path):
    bits = [1, 1, 1, 1, 1, 0]
    guesses = [1] * 6
    out, server = drive(node, tmp_path, "predict", {"guesses": guesses}, [[b] for b in bits])
    hits = 0
    streak = 0
    for k, snap in enumerate(out["snaps"], start=1):
        hit = bits[k - 1] == guesses[k - 1]
        hits += hit
        streak = streak + 1 if hit else 0
        assert snap["result"].startswith(f"Bit {bits[k - 1]}, you guessed 1: {'Hit' if hit else 'Miss'}"), snap["label"]
        assert snap["gsLeft"] == f"{hits}/{k} hits"
        assert snap["gsCenter"] == f"{hits / k * 100:.1f}% hit rate"
        assert snap["gsRight"] == f"{z_score(hits, k):.2f}σ"
        want = streak_text(streak)
        assert (want in snap["guessStreak"]) if want else snap["guessStreak"] == "", snap["label"]
        assert snap["alerts"] == []
    assert server["stats"]["total"]["predict"]["hits"] == 5


def test_perceive_hides_the_bit_until_the_guess_then_renders_the_result(node, tmp_path):
    out, server = drive(node, tmp_path, "perceive", {"guess": 1}, [[1]])
    before, prepared, revealed = out["snaps"]
    assert before["prepareHidden"] is False and before["guessButtonsHidden"] is True
    assert prepared["prepareHidden"] is True and prepared["guessButtonsHidden"] is False
    assert prepared["result"] == "" and "hidden" in prepared["prompt"]
    assert revealed["result"].startswith("Bit 1, you guessed 1: Hit")
    assert revealed["gsLeft"] == "1/1 hits"
    assert revealed["alerts"] == []
    assert server["stats"]["total"]["perceive"]["hits"] == 1


def test_a_server_error_is_shown_verbatim(node, tmp_path):
    """A press whose job fails: the error reaches the alert as code: detail, no default value shown."""
    out, _ = drive(node, tmp_path, "press_error", {"target": 1})  # no scripted bits: the job raises
    snap = out["snaps"][0]
    assert snap["alerts"] == ["provider_error: ProviderError: no scripted bits left"]
    assert snap["big"] == "+0.000%" and snap["shots"] == 0


def test_a_mid_stream_failure_is_shown_once(node, tmp_path):
    """The failure arrives as an SSE error event and again as session.stream.error: one alert."""
    out, server = drive_stream(node, tmp_path, "stream_error", {"target": 1}, [[1] * 20, ProviderError("device error")])
    snap = out["snaps"][0]
    assert snap["alerts"] == ["stream / provider_error: ProviderError: device error"]
    assert server["stream"]["running"] is False
    assert server["stream"]["error"] == "provider_error: ProviderError: device error"
    assert server["stream"]["stopped_reason"] is None  # a failure is not a limit stop
    assert snap["shots"] == 20


def test_buffered_bits_are_labelled_buffered_in_the_footer(node, tmp_path):
    """Design requirement: buffered bits never pass as live ones. Fails if the footer says 'live'."""
    out, _ = drive(node, tmp_path, "press", {"presses": 1, "target": 1, "source": "buffered"}, [[1]])
    prov = out["snaps"][0]["provenance"]
    assert " · buffered · " in prov and "live" not in prov, prov


def test_buffered_predict_says_the_bit_was_measured_before_the_guess(node, tmp_path):
    out, _ = drive(node, tmp_path, "predict", {"guesses": [1], "source": "buffered"}, [[1]])
    prov = out["snaps"][0]["provenance"]
    assert " · buffered · " in prov and prov.endswith("measured before guess"), prov


def test_live_predict_does_not_claim_a_prior_measurement(node, tmp_path):
    out, _ = drive(node, tmp_path, "predict", {"guesses": [1]}, [[1]])
    prov = out["snaps"][0]["provenance"]
    assert " · live · " in prov and "measured before guess" not in prov, prov


def test_ball_and_side_labels_use_the_5_percent_track_inside_the_threshold(node, tmp_path):
    """25 presses with 13 ones end at +4.0 %: inside the track, so the ball is not clamped.
    A track threshold other than 5 moves ballTarget or the side labels."""
    bits = [1, 0] * 12 + [1]
    out, _ = drive(node, tmp_path, "press", {"presses": len(bits), "target": 1}, [[b] for b in bits])
    last = out["snaps"][-1]
    assert last["right"] == "+4.00%" and last["ballTarget"] == pytest.approx(90.0)
    assert last["trackLeft"] == "-5.0%" and last["trackRight"] == "+5.0%"
    assert 0 < last["ballTarget"] < 100


def drive_slow_press(node, tmp_path, params):
    """Session A's press is held by the server for delay_ms; the harness exits and starts session B."""
    app = scripted_app(tmp_path, [[1]], delays=[params["delay_ms"] / 1000])  # in the provider thread
    srv = Server(app)
    try:
        r = subprocess.run([node, str(HARNESS), srv.base, "busy_after_exit", json.dumps(params)],
                           capture_output=True, text=True, timeout=90)
        assert r.returncode == 0, f"harness failed:\n{r.stderr}"
        return json.loads(r.stdout)
    finally:
        srv.stop()


def test_slow_request_of_an_exited_session_does_not_dim_the_next_session(node, tmp_path):
    """C2: A's request is slow (busy shown), the user exits and starts B; when A resolves B is not busy."""
    out = drive_slow_press(node, tmp_path, {"delay_ms": 1500})
    c = out["cases"]
    assert c["busyWhileSlow"] is True
    assert out["sidA"] != out["sidB"]
    for when in ("afterExit", "afterOpenB", "timerWindow", "afterAResolved"):
        assert "busy" not in c[when].split(), (when, c[when])


def test_busy_timer_firing_after_exit_adds_nothing(node, tmp_path):
    """C2: exit and start B inside the 250 ms window; A's timer fires after exit and must not dim B."""
    out = drive_slow_press(node, tmp_path, {"delay_ms": 1500, "exit_before_timer": True})
    c = out["cases"]
    assert "busyWhileSlow" not in c
    assert "busy" not in c["timerWindow"].split(), c["timerWindow"]
    assert "busy" not in c["afterAResolved"].split(), c["afterAResolved"]


# ---------------------------------------------------------------- stream limits
def qpu_estimate(shots, jobs):
    return (2 + 0.00035 * shots) * jobs  # the server's formula (GET /api/budget qpu_estimate), computed here


def test_setup_asks_for_stream_limits_validates_them_and_sends_both(node, tmp_path):
    out, server = drive(node, tmp_path, "stream_setup", {"provider": "ibm"})
    # shown for Stream only; Max IBM jobs for ibm only
    assert out["hidden"] == {"keyPress": True, "stream": False, "jobsRowSim": True, "jobsRowIbm": False, "guess": True}
    assert out["defaults"] == {"seconds": "30", "jobs": "5"}
    assert f"up to {qpu_estimate(20, 5):.1f} s" in out["estimateDefault"] and "5 jobs of 20 shots" in out["estimateDefault"]
    assert f"up to {qpu_estimate(20, 7):.1f} s" in out["estimateSeven"]
    assert f"up to {qpu_estimate(2000, 7):.1f} s" in out["estimateBigShots"]
    assert out["estimateBuffered"].startswith("Buffered streams submit no jobs")
    # bad values: an alert naming the field with the value, no session
    secs = "Stream duration must be a whole number of at least 1, got"
    jobs = "Max IBM jobs must be a whole number of at least 1, got"
    assert [a.split('"')[0].strip() for a in out["alerts"]] == [secs] * 3 + [jobs] * 2
    assert [a.split('"')[1] for a in out["alerts"]] == ["0", "abc", "2.5", "0", ""]
    assert out["focusHiddenAfterBad"] is True
    # the valid Start sent both fields; the server stored them
    assert server["provider"] == "ibm" and server["stream_seconds"] == 45 and server["stream_max_jobs"] == 7


def test_stream_countdown_runs_and_the_server_limit_flips_the_view_to_stopped(node, tmp_path):
    batches = [[1] * 20 for _ in range(10)]
    out, server = drive_stream(node, tmp_path, "stream_limit", {"seconds": 2}, batches)
    running = out["running"][0]
    assert running["barHidden"] is False and running["stopHidden"] is False
    assert running["barText"] in ("Stream: 2 s left", "Stream: 1 s left"), running["barText"]
    assert "jobs" not in running["barText"]  # sim: no QPU jobs to count
    assert len(out["seen"]) >= 2  # it counted down
    stopped = out["stopped"]
    assert stopped["barText"] == "Stream stopped: 2 s limit reached"
    assert "limit" in stopped["barClass"].split() and stopped["stopHidden"] is True
    assert stopped["alerts"] == []  # a limit is not an error: no red alert
    assert "stream stopped" in stopped["provenance"]
    assert out["timerCleared"] is True
    assert server["stream"]["running"] is False and server["stream"]["stopped_reason"].startswith("limit_seconds:")
    assert server["stream"]["error"] is None
    assert server["stats"]["total"]["shots"] >= 20 and server["stats"]["total"]["shots"] % 20 == 0


def test_ibm_live_stream_shows_jobs_used_and_the_job_limit_message(node, tmp_path):
    out, server = drive(node, tmp_path, "stream_limit", {"provider": "ibm", "seconds": 60, "max_jobs": 2})
    assert any(t.endswith(" · jobs 1/2") for t in out["seen"]), out["seen"]
    assert any(t.endswith(" · jobs 2/2") for t in out["seen"]), out["seen"]
    assert all(" s left · jobs " in t for t in out["seen"])
    assert out["stopped"]["barText"] == "Stream stopped: 2 job limit reached"
    assert out["stopped"]["alerts"] == []
    assert server["stream"]["stopped_reason"].startswith("limit_jobs:") and server["stream"]["jobs_used"] == 2


def test_the_countdown_ticks_by_itself_between_server_events(node, tmp_path):
    out, _ = drive_stream(node, tmp_path, "stream_countdown_idle", {"seconds": 60}, [[1] * 20])  # one batch, then silence
    left = lambda s: int(s["barText"].split()[1])
    assert left(out["first"]) in (59, 60)
    assert left(out["first"]) - left(out["later"]) in (2, 3), (out["first"]["barText"], out["later"]["barText"])


def test_the_stop_control_stops_the_stream_without_a_limit_message(node, tmp_path):
    out, server = drive_stream(node, tmp_path, "stream_stop_button", {"seconds": 60}, [[1] * 20 for _ in range(10)])
    assert out["running"]["barHidden"] is False and out["running"]["stopHidden"] is False
    assert out["running"]["barText"].startswith("Stream: ") and out["running"]["barText"].endswith(" s left")
    stopped = out["stopped"]
    assert stopped["stopHidden"] is True and stopped["barHidden"] is True and stopped["alerts"] == []
    assert server["stream"]["running"] is False and server["stream"]["stopped_reason"] is None
