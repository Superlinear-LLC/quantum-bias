"""The web UI: static files are served, parse, and their pure logic (logic.js) behaves.

Node is only needed for these tests, never by the app: they skip without it.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "quantum_lite" / "static"


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; UI tests need it (the app itself does not)")
    return node


def node_eval(script: str):
    out = subprocess.run([_node(), "-e", f"const L=require({json.dumps(str(STATIC / 'logic.js'))});{script}"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_every_static_js_file_parses():
    node = _node()
    files = sorted(STATIC.rglob("*.js"))
    assert {f.name for f in files} >= {"app.js", "logic.js"}
    for f in files:
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f.name}: {r.stderr}"


def test_ui_is_served_with_its_assets(client):
    html = client.get("/").text
    assert "Start Training" in html and 'id="focus"' in html
    for path in ("/app.js", "/logic.js", "/style.css", "/fonts/Geist-Variable.woff2",
                 "/fonts/GeistMono-Variable.woff2", "/fonts/LICENSE.TXT"):
        assert client.get(path).status_code == 200, path


def test_static_files_are_revalidated_not_heuristically_cached(client):
    for path in ("/", "/app.js", "/style.css", "/logic.js"):
        r = client.get(path)
        assert r.headers["cache-control"] == "no-cache", path
        assert r.headers.get("etag"), path
        assert client.get(path, headers={"If-None-Match": r.headers["etag"]}).status_code == 304, path


def test_ui_references_no_external_host():
    for name in ("index.html", "style.css", "app.js", "logic.js"):
        text = (STATIC / name).read_text()
        assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")
        assert "https://" not in text, name


# --- ball position (useBallPosition.ts) ------------------------------------------------------
def test_ball_position_maps_threshold_edges_and_clamps():
    out = node_eval("""
      const r = (b) => L.ballPosition(b, 5);
      console.log(JSON.stringify([r(-5), r(0), r(5), r(2.5), r(-1.25), r(66.667), r(-20), r(5.0001)]));
    """)
    pos = [(round(o["position"], 6), o["isClamped"]) for o in out]
    assert pos == [(0, False), (50, False), (100, False), (75, False), (37.5, False),
                   (100, True), (0, True), (100, True)]


def test_ball_position_rejects_a_non_positive_threshold():
    node = _node()
    r = subprocess.run([node, "-e", f"require({json.dumps(str(STATIC / 'logic.js'))}).ballPosition(1, 0)"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "threshold must be > 0" in r.stderr


def test_target_zone_per_target():
    out = node_eval("console.log(JSON.stringify([L.inTargetZone(5,1,5), L.inTargetZone(4.99,1,5),"
                    "L.inTargetZone(-5,0,5), L.inTargetZone(5,0,5), L.inTargetZone(-9,1,5)]))")
    assert out == [True, False, True, False, False]


# --- ball spring ----------------------------------------------------------
def test_spring_converges_without_overshoot_and_at_any_frame_rate():
    out = node_eval("""
      function run(dt, secs) {
        let s = {x: 0, v: 0}, peak = 0, t = 0, settledAt = null;
        while (t < secs) {
          s = L.springStep(s, 100, dt); t += dt; peak = Math.max(peak, s.x);
          if (settledAt === null && L.springSettled(s, 100)) settledAt = t;
        }
        return {x: s.x, peak, settledAt};
      }
      console.log(JSON.stringify({a: run(1/60, 15), b: run(1/20, 15), c: run(0.05, 15)}));
    """)
    for k in ("a", "b", "c"):
        assert out[k]["x"] == pytest.approx(100, abs=0.01)
        assert out[k]["peak"] <= 100.0001  # overdamped (zeta ~ 1.94): never overshoots
        assert out[k]["settledAt"] is not None
    assert 3 < out["a"]["settledAt"] < 12  # slow, smooth: not a snap, not endless
    assert out["a"]["x"] == pytest.approx(out["b"]["x"], abs=0.01)


def test_spring_is_monotone_and_moves_at_the_start():
    out = node_eval("""
      let s = {x: 50, v: 0}; const xs = [];
      for (let i = 0; i < 120; i++) { s = L.springStep(s, 0, 1/60); xs.push(s.x); }
      console.log(JSON.stringify({xs}));
    """)
    xs = out["xs"]
    assert xs[0] < 50 and all(b <= a + 1e-9 for a, b in zip(xs, xs[1:]))


def test_spring_at_rest_on_target_stays_put_and_negative_dt_raises():
    out = node_eval("console.log(JSON.stringify(L.springStep({x: 40, v: 0}, 40, 0.016)))")
    assert out == {"x": 40, "v": 0}
    r = subprocess.run([_node(), "-e", f"require({json.dumps(str(STATIC / 'logic.js'))}).springStep({{x:0,v:0}},1,-1)"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "dt must be >= 0" in r.stderr


# --- streaks ---------------------------------------------------------------------------------
def test_next_streak_counts_target_bits_and_resets_on_a_miss():
    out = node_eval("""
      let s = 0; const seq = [];
      for (const b of [1, 1, 0, 1, 1, 1, 0]) { s = L.nextStreak(s, b, 1); seq.push(s); }
      let z = 0; const seq0 = [];
      for (const b of [0, 0, 1, 0]) { z = L.nextStreak(z, b, 0); seq0.push(z); }
      console.log(JSON.stringify({seq, seq0}));
    """)
    assert out["seq"] == [1, 2, 0, 1, 2, 3, 0]
    assert out["seq0"] == [1, 2, 0, 1]


@pytest.mark.parametrize("prev,tail_bit,tail_run,shots,target,expected", [
    (3, 1, 10, 10, 1, 13),   # whole batch is target: streak continues through it
    (3, 1, 4, 10, 1, 4),     # a miss inside the batch: only the final run counts
    (3, 0, 10, 10, 1, 0),    # whole batch is the other bit
    (3, 0, 2, 10, 1, 0),     # ends on a miss
    (3, 0, 10, 10, 0, 13),   # target 0
    (7, 1, 1, 1, 1, 8),      # single-shot batch
])
def test_batch_streak_follows_the_trailing_run(prev, tail_bit, tail_run, shots, target, expected):
    got = node_eval(f"console.log(JSON.stringify(L.batchStreak({prev},{tail_bit},{tail_run},{shots},{target})))")
    assert got == expected


def test_batch_streak_matches_bit_by_bit_counting():
    out = node_eval("""
      let bad = 0;
      for (let t = 0; t < 400; t++) {
        const n = 1 + (t % 9), bits = Array.from({length: n}, (_, i) => ((t * 7 + i * i * 3 + (i >> 1)) >> 1) & 1);
        const target = t & 1, prev = t % 5;
        let s = prev; for (const b of bits) s = L.nextStreak(s, b, target);
        const last = bits[n - 1]; let run = 0; for (let i = n - 1; i >= 0 && bits[i] === last; i--) run++;
        if (L.batchStreak(prev, last, run, n, target) !== s) bad++;
      }
      console.log(bad);
    """)
    assert out == 0


def test_batch_streak_rejects_an_impossible_tail():
    r = subprocess.run([_node(), "-e", f"require({json.dumps(str(STATIC / 'logic.js'))}).batchStreak(0,1,5,3,1)"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "bad tail run" in r.stderr


# --- fire tier and streak label --------------------------------
def test_fire_tier_boundaries_at_threshold_5():
    out = node_eval("console.log(JSON.stringify([0,4,5,9,10,14,15,19,20,99].map(n => L.fireTier(n, 5))))")
    assert out == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_fire_tier_honours_a_different_threshold_below_ten():
    out = node_eval("console.log(JSON.stringify([2,3,10].map(n => L.fireTier(n, 3))))")
    assert out == [0, 1, 2]


def test_streak_labels_match_the_tiers():
    out = node_eval("console.log(JSON.stringify([4,5,9,10,15,20,27].map(n => L.streakLabel(n, 5))))")
    assert out[0] is None
    assert [(o["label"], o["color"], o["text"]) for o in out[1:]] == [
        ("Lucky", "green", "5 in a row — Lucky"),
        ("Lucky", "green", "9 in a row — Lucky"),
        ("Epic", "blue", "10 in a row — Epic"),
        ("Mythic", "purple", "15 in a row — Mythic"),
        ("LEGENDARY", "orange", "20 in a row — LEGENDARY"),
        ("LEGENDARY", "orange", "27 in a row — LEGENDARY"),
    ]
    assert [o["emoji"] for o in out[1:]] == ["\U0001F340", "\U0001F340", "\U0001F48E", "\U0001F49C", "\U0001F525", "\U0001F525"]
    assert out[1]["key"] == 1 and out[2]["key"] == 1 and out[3]["key"] == 2  # re-animates per 5 in a row


# --- formatting ------------------------------------------------------------------------------
def test_percent_and_sigma_formatting():
    out = node_eval("""
      console.log(JSON.stringify([L.fmtSignedPct(0, 3), L.fmtSignedPct(66.66666, 3), L.fmtSignedPct(-1.5, 2),
        L.fmtSignedPct(100, 2), L.fmtThreshold(5), L.fmtSigma(1.634), L.fmtSigma(-0.004), L.fmtSigma(0)]));
    """)
    assert out == ["+0.000%", "+66.667%", "-1.50%", "+100.00%", "5.0%", "1.63σ", "-0.00σ", "0.00σ"]


def test_bias_tone_zone_beats_sign():
    out = node_eval("console.log(JSON.stringify([L.biasTone(9,true), L.biasTone(1,false), L.biasTone(-1,false), L.biasTone(0,false)]))")
    assert out == ["zone", "pos", "neg", "zero"]


# --- qubit rules -----------------------------------------------------------------------------
def test_majority_qubits_are_odd_only_and_ranges_hold():
    out = node_eval("""
      console.log(JSON.stringify({
        none: L.qubitRange('none'), xor: L.qubitRange('xor'), maj: L.qubitRange('majority'),
        norm: [L.normalizeQubits(4, 'majority'), L.normalizeQubits(16, 'majority'), L.normalizeQubits(0, 'xor'),
               L.normalizeQubits(99, 'xor'), L.normalizeQubits(7, 'none'), L.normalizeQubits(2, 'majority')],
        step: [L.stepQubits(3, 1, 'majority'), L.stepQubits(1, -1, 'majority'), L.stepQubits(15, 1, 'majority'),
               L.stepQubits(5, 1, 'xor'), L.stepQubits(16, 1, 'xor')],
      }));
    """)
    assert out["none"] == {"min": 1, "max": 1, "step": 1}
    assert out["xor"] == {"min": 1, "max": 16, "step": 1}
    assert out["maj"] == {"min": 1, "max": 15, "step": 2}
    assert out["norm"] == [3, 15, 1, 16, 1, 1]
    assert out["step"] == [5, 1, 15, 6, 16]
    assert all(n % 2 == 1 for n in out["norm"][:2] + [out["norm"][5]])


def test_pool_available_sums_matching_pools_only():
    out = node_eval("""
      const pools = [
        {provider: 'sim', implementation: 'aer_ideal', n_qubits: 5, available_shots: 100},
        {provider: 'sim', implementation: 'aer_fake_toronto', n_qubits: 5, available_shots: 7},
        {provider: 'sim', implementation: 'aer_ideal', n_qubits: 3, available_shots: 50},
        {provider: 'ibm', implementation: 'ibm_torino', n_qubits: 5, available_shots: 1000},
      ];
      console.log(JSON.stringify([L.poolAvailable(pools, 'sim', 'aer_ideal', 5).shots,
        L.poolAvailable(pools, 'sim', null, 5).shots, L.poolAvailable(pools, 'ibm', null, 5).shots,
        L.poolAvailable(pools, 'ibm', null, 2).shots]));
    """)
    assert out == [100, 107, 1000, 0]


def test_pool_line_does_not_sum_devices_when_the_implementation_is_unknown():
    out = node_eval("""
      const f = (n) => String(n);
      const pools = [
        {provider: 'ibm', implementation: 'ibm_torino', n_qubits: 5, available_shots: 1000},
        {provider: 'ibm', implementation: 'ibm_fez', n_qubits: 5, available_shots: 200},
        {provider: 'ibm', implementation: 'ibm_torino', n_qubits: 3, available_shots: 40},
      ];
      console.log(JSON.stringify([
        L.poolLine(pools, 'ibm', null, 5, f), L.poolLine(pools, 'ibm', 'ibm_fez', 5, f),
        L.poolLine(pools, 'ibm', null, 3, f), L.poolLine(pools, 'ibm', null, 2, f)]));
    """)
    two, pinned, one, none = out
    assert "1200" not in two and "ready" not in two
    assert "ibm_torino 1000" in two and "ibm_fez 200" in two and "device is chosen at the first job" in two
    assert pinned == "Pool ibm/ibm_fez, 5 qubits: 200 shots ready"
    assert one == "Pool ibm, 3 qubits: ibm_torino 40 shots; the device is chosen at the first job"
    assert "ready" not in one
    assert none == "Pool ibm, 2 qubits: empty. Prefetch before starting."


def test_sigma_of_a_missing_or_non_finite_value_is_an_error_not_zero():
    node = _node()
    for bad in ("undefined", "null", "NaN", "'1.0'"):
        r = subprocess.run([node, "-e", f"require({json.dumps(str(STATIC / 'logic.js'))}).fmtSigma({bad})"],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "sigma must be a finite number" in r.stderr, bad


# --- session state reducers (the wiring app.js uses) ----------------------------------------
_REDUCER_SETUP = """
const mk = (sid, kind, target) => ({session: sid === null ? null : {id: sid}, kind, target, bias: 0, z: 0, n: 0,
  streak: 0, streaming: false, trial: null, result: null, guessStats: null, statsShots: 0, prov: null});
const stats = (shots, bias, z, extra) => ({total: Object.assign({shots, aggregated: {bias_pct: bias, z, n: shots}}, extra || {})});
"""


def test_press_reducer_counts_the_target_streak_and_applies_stats():
    out = node_eval(_REDUCER_SETUP + """
      const F = mk('A', 'key_press', 1);
      const prov = {job_id: 'j1'};
      L.applyPress(F, 'A', {bit: 1, provenance: prov, stats: stats(1, 100, 1)});
      L.applyPress(F, 'A', {bit: 1, provenance: prov, stats: stats(2, 100, 1.41)});
      const two = F.streak;
      L.applyPress(F, 'A', {bit: 0, provenance: prov, stats: stats(3, 33.3, 0.58)});
      console.log(JSON.stringify({two, after0: F.streak, bias: F.bias, z: F.z, n: F.n, prov: F.prov}));
    """)
    assert out == {"two": 2, "after0": 0, "bias": 33.3, "z": 0.58, "n": 3, "prov": {"job_id": "j1"}}
    out = node_eval(_REDUCER_SETUP + """
      const F = mk('A', 'key_press', 0);
      L.applyPress(F, 'A', {bit: 0, provenance: {}, stats: stats(1, -100, -1)});
      console.log(JSON.stringify(F.streak));
    """)
    assert out == 1


def test_a_response_from_an_exited_session_never_writes_into_the_next_one():
    """F1: session A's late press/prepare/batch/stats must leave session B untouched."""
    out = node_eval(_REDUCER_SETUP + """
      const B = mk('B', 'key_press', 1);          // A exited, B opened fresh
      const late = {bit: 1, provenance: {job_id: 'A-job'}, stats: stats(4, 100, 2)};
      const applied = [L.applyPress(B, 'A', late), L.applyStats(B, 'A', late.stats),
        L.applyBatch(B, 'A', {tail_bit: 1, tail_run: 40, shots: 40, provenance: {job_id: 'A-job'}}),
        L.applyStream(B, 'A', {running: true, seconds_left: 5}, 0), L.applyPrepare(B, 'A', {trial_id: 'tA'}),
        L.applyGuess(B, 'A', {bit: 1, guess: 1, hit: true, provenance: {}, stats: stats(4, 100, 2)}),
        L.applyReveal(B, 'A', {bit: 1, guess: 1, hit: true, provenance: {}, stats: stats(4, 100, 2)})];
      // B's own first press still lands: statsShots was not raised by A
      const own = L.applyPress(B, 'B', {bit: 0, provenance: {job_id: 'B-job'}, stats: stats(1, -100, -1)});
      console.log(JSON.stringify({applied, own, n: B.n, shots: B.statsShots, bias: B.bias, prov: B.prov,
        trial: B.trial, streaming: B.streaming, streak: B.streak}));
    """)
    assert out == {"applied": [False] * 7, "own": True, "n": 1, "shots": 1, "bias": -100,
                   "prov": {"job_id": "B-job"}, "trial": None, "streaming": False, "streak": 0}


def test_reducers_drop_everything_when_no_session_is_open():
    out = node_eval(_REDUCER_SETUP + """
      const F = mk(null, 'key_press', 1);
      console.log(JSON.stringify([L.applyPress(F, 'A', {bit: 1, provenance: {}, stats: stats(1, 1, 1)}),
        L.applyStats(F, undefined, stats(1, 1, 1)), F.n]));
    """)
    assert out == [False, False, 0]


def test_stats_only_grow_so_out_of_order_responses_are_ignored():
    out = node_eval(_REDUCER_SETUP + """
      const F = mk('A', 'key_press', 1);
      const a = L.applyStats(F, 'A', stats(5, 20, 0.5)), b = L.applyStats(F, 'A', stats(3, 10, 0.2));
      console.log(JSON.stringify([a, b, F.n, F.bias]));
    """)
    assert out == [True, False, 5, 20]


def test_stream_batch_reducer_continues_or_resets_the_streak():
    out = node_eval(_REDUCER_SETUP + """
      const F = mk('A', 'stream', 1);
      const ev = (bit, run, shots) => ({tail_bit: bit, tail_run: run, shots, provenance: {job_id: 'j'}});
      const seq = [];
      for (const e of [ev(1, 40, 40), ev(1, 40, 40), ev(1, 3, 40), ev(0, 5, 40), ev(1, 40, 40)]) { L.applyBatch(F, 'A', e); seq.push(F.streak); }
      const t0 = mk('A', 'stream', 0); L.applyBatch(t0, 'A', ev(0, 7, 40));
      console.log(JSON.stringify([seq, t0.streak]));
    """)
    assert out == [[40, 80, 3, 0, 40], 7]


def test_guess_reducers_record_the_result_and_the_server_streak():
    out = node_eval(_REDUCER_SETUP + """
      const g = (hits, trials, streak) => ({hits, trials, streak, hit_rate: hits / trials, z_hits: 0.5});
      const F = mk('A', 'perceive', 1);
      L.applyPrepare(F, 'A', {trial_id: 't1', source: 'live'});
      const hasTrial = F.trial.trial_id;
      L.applyReveal(F, 'A', {bit: 1, guess: 1, hit: true, measured_before_guess: true, provenance: {job_id: 'j'},
        stats: stats(1, 100, 1, {perceive: g(1, 1, 1)})});
      console.log(JSON.stringify({hasTrial, trial: F.trial, result: F.result, streak: F.streak, g: F.guessStats.hits}));
    """)
    assert out == {"hasTrial": "t1", "trial": None, "streak": 1, "g": 1,
                   "result": {"bit": 1, "guess": 1, "hit": True, "before": True}}
    node = _node()
    r = subprocess.run([node, "-e", f"const L=require({json.dumps(str(STATIC / 'logic.js'))});" + _REDUCER_SETUP +
                        "L.applyStats(mk('A','predict',1),'A',stats(1,0,0,{}))"], capture_output=True, text=True)
    assert r.returncode != 0 and "stats.total.predict missing" in r.stderr


# --- job state line and QPU estimate ---------------------------------------------------------
def test_job_status_text_shows_state_queue_position_and_elapsed():
    out = node_eval("""
      const t0 = Date.parse('2026-01-01T00:00:00+00:00');
      const j = (o) => Object.assign({id: 'loc', job_id: 'ibm-1', state: 'queued', queue_position: null,
        submitted_at: '2026-01-01T00:00:00+00:00', finished_at: null}, o);
      console.log(JSON.stringify([L.jobStatusText(j({}), t0 + 12999), L.jobStatusText(j({queue_position: 4, state: 'queued'}), t0 + 61000),
        L.jobStatusText(j({job_id: null, state: 'submitting'}), t0 - 5000),
        L.jobStatusText(j({state: 'done', finished_at: '2026-01-01T00:00:07+00:00'}), t0 + 999999)]));
    """)
    assert out == ["job ibm-1 queued, queue position not reported, elapsed 12s",
                   "job ibm-1 queued, queue position 4, elapsed 61s",
                   "job loc submitting, queue position not reported, elapsed 0s",
                   "job ibm-1 done, queue position not reported, elapsed 7s"]


def test_job_status_text_rejects_a_job_without_a_valid_time():
    node = _node()
    r = subprocess.run([node, "-e", f"require({json.dumps(str(STATIC / 'logic.js'))}).jobStatusText("
                        "{id:'x',state:'queued',submitted_at:undefined},0)"], capture_output=True, text=True)
    assert r.returncode != 0 and "no valid submitted_at" in r.stderr


def test_qpu_estimate_uses_the_server_formula(client):
    est = client.get("/api/budget").json()["qpu_estimate"]
    assert est["is_estimate"] is True and est["formula"] == "2 + 0.00035 * shots"
    out = node_eval(f"console.log(JSON.stringify(L.estimateQpuSeconds({json.dumps(est)}, 20000)))")
    assert out == pytest.approx(9.0)


# ---------------------------------------------------------------- stream limits: countdown text, validation
_BAR_SETUP = """
  const mk = (provider, mode) => ({session: {id: 'A', provider, mode}, streaming: false, streamInfo: null, deadlineAt: null});
  const stream = (o) => ({running: false, error: null, stopped_reason: null, seconds_limit: 30, seconds_left: null,
                          jobs_limit: null, jobs_used: 0, ...o});
"""


def test_countdown_text_counts_whole_seconds_up_and_shows_jobs_for_ibm_only():
    out = node_eval(_BAR_SETUP + """
      const sim = mk('sim', 'live'), ibm = mk('ibm', 'live'), ibmBuf = mk('ibm', 'buffered');
      L.applyStream(sim, 'A', stream({running: true, seconds_left: 12.3}), 1000);
      L.applyStream(ibm, 'A', stream({running: true, seconds_left: 12.3, jobs_limit: 5, jobs_used: 2}), 1000);
      L.applyStream(ibmBuf, 'A', stream({running: true, seconds_left: 12.3, jobs_limit: 5, jobs_used: 2}), 1000);
      console.log(JSON.stringify([L.streamBar(sim, 1000).text, L.streamBar(sim, 1000 + 12299).text,
        L.streamBar(sim, 1000 + 99999).text, L.streamBar(ibm, 1000).text, L.streamBar(ibmBuf, 1000).text,
        L.streamBar(ibm, 1000).tone]));
    """)
    assert out == ["Stream: 13 s left", "Stream: 1 s left", "Stream: 0 s left",
                   "Stream: 13 s left · jobs 2/5", "Stream: 13 s left · jobs 2/5", "run"]


def test_limit_stop_text_names_the_limit_and_is_not_an_error_tone():
    out = node_eval(_BAR_SETUP + """
      const a = mk('sim', 'live'), b = mk('ibm', 'live'), c = mk('sim', 'live');
      L.applyStream(a, 'A', stream({stopped_reason: 'limit_seconds: over', seconds_limit: 45}), 0);
      L.applyStream(b, 'A', stream({stopped_reason: 'limit_jobs: 3 used', jobs_limit: 3, jobs_used: 3}), 0);
      L.applyStream(c, 'A', stream({}), 0);
      let bad = null;
      try { const d = mk('sim', 'live'); L.applyStream(d, 'A', stream({stopped_reason: 'weird'}), 0); L.streamBar(d, 0); }
      catch (e) { bad = e.message; }
      console.log(JSON.stringify([L.streamBar(a, 0), L.streamBar(b, 0), L.streamBar(c, 0), bad]));
    """)
    assert out == [{"text": "Stream stopped: 45 s limit reached", "tone": "limit"},
                   {"text": "Stream stopped: 3 job limit reached", "tone": "limit"}, None,
                   "unknown stream stop reason weird"]


def test_stream_limit_fields_must_be_positive_whole_numbers():
    out = node_eval("""
      const t = (s, j, ibm) => { try { return L.streamLimits(s, j, ibm); } catch (e) { return e.message; } };
      console.log(JSON.stringify([t('30', '5', true), t('30', '', false), t('0', '5', true), t('1.5', '5', true),
        t('-3', '5', true), t(' 7 ', '5', false), t('30', '0', true), t('30', 'x', true), t('', '5', true)]));
    """)
    assert out == [{"stream_seconds": 30, "stream_max_jobs": 5}, {"stream_seconds": 30},
                   'Stream duration must be a whole number of at least 1, got "0"',
                   'Stream duration must be a whole number of at least 1, got "1.5"',
                   'Stream duration must be a whole number of at least 1, got "-3"',
                   {"stream_seconds": 7},
                   'Max IBM jobs must be a whole number of at least 1, got "0"',
                   'Max IBM jobs must be a whole number of at least 1, got "x"',
                   'Stream duration must be a whole number of at least 1, got ""']


def test_stream_qpu_estimate_is_max_jobs_times_the_per_job_estimate():
    out = node_eval("""
      const est = {base_seconds: 2, seconds_per_shot: 0.00035};
      console.log(JSON.stringify([L.streamQpuEstimate(est, 20, 5), L.streamQpuEstimate(est, 2000, 3)]));
    """)
    assert out == [pytest.approx(10.035), pytest.approx(8.1)]
