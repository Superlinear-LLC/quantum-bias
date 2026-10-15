// Pure UI logic, kept free of the DOM so tests can run it under node.
//
// Ball position mapping and clamping, fire tiers and streak message tiers, and the ball spring.
(function (root) {
  const DEFAULT_THRESHOLD_PCT = 5; // difficulty level 1
  const FIRE_THRESHOLD = 5;
  // Ball spring: stiffness 40, damping 30, mass 1.5 (damping ratio ~1.94,
  // so overdamped: no overshoot).
  const SPRING = { stiffness: 40, damping: 30, mass: 1.5 };
  const REST_DELTA = 0.01; // percent of the track
  const REST_SPEED = 0.01;

  // -threshold -> 0 %, 0 -> 50 %, +threshold -> 100 %; values beyond the threshold are clamped.
  function ballPosition(biasPct, thresholdPct) {
    if (!(thresholdPct > 0)) throw new RangeError(`threshold must be > 0, got ${thresholdPct}`);
    const clamped = Math.max(-thresholdPct, Math.min(thresholdPct, biasPct));
    return {
      position: ((clamped + thresholdPct) / (2 * thresholdPct)) * 100,
      isClamped: Math.abs(biasPct) > thresholdPct,
    };
  }

  function inTargetZone(biasPct, target, thresholdPct) {
    return target === 1 ? biasPct >= thresholdPct : biasPct <= -thresholdPct;
  }

  // One step of a damped spring x'' = (-k (x - target) - c x') / m. Semi-implicit Euler in
  // sub-steps of at most 1/240 s, stable for any frame time. Returns a new state.
  function springStep(state, target, dt, params) {
    const p = params || SPRING;
    if (!(dt >= 0)) throw new RangeError(`dt must be >= 0, got ${dt}`);
    let { x, v } = state;
    const steps = Math.max(1, Math.ceil(dt * 240));
    const h = dt / steps;
    for (let i = 0; i < steps; i++) {
      const a = (-p.stiffness * (x - target) - p.damping * v) / p.mass;
      v += a * h;
      x += v * h;
    }
    return { x, v };
  }

  function springSettled(state, target) {
    return Math.abs(state.x - target) < REST_DELTA && Math.abs(state.v) < REST_SPEED;
  }

  // Streak of consecutive aggregated bits equal to the target (Affect).
  function nextStreak(prev, bit, target) {
    return bit === target ? prev + 1 : 0;
  }

  // Streak after a stream batch. The server sends only the trailing run of the batch's
  // aggregated bits: (tail_bit, tail_run) of `shots` bits. If the whole batch is one run of the
  // target, the streak continues through it; otherwise it is the run at the end, or 0.
  function batchStreak(prev, tailBit, tailRun, shots, target) {
    if (!(tailRun >= 1 && tailRun <= shots)) throw new RangeError(`bad tail run ${tailRun} of ${shots}`);
    if (tailBit !== target) return 0;
    return tailRun === shots ? prev + shots : tailRun;
  }

  function fireTier(streak, fireThreshold) {
    const t = fireThreshold === undefined ? FIRE_THRESHOLD : fireThreshold;
    return streak >= 20 ? 4 : streak >= 15 ? 3 : streak >= 10 ? 2 : streak >= t ? 1 : 0;
  }

  // Message under the percentage; null below the fire threshold. Labels, emoji and colour
  // names are fixed by the design.
  function streakLabel(streak, fireThreshold) {
    const t = fireThreshold === undefined ? FIRE_THRESHOLD : fireThreshold;
    if (streak < t) return null;
    const tier = streak >= 20 ? { label: "LEGENDARY", color: "orange", emoji: "\u{1F525}" }
      : streak >= 15 ? { label: "Mythic", color: "purple", emoji: "\u{1F49C}" }
      : streak >= 10 ? { label: "Epic", color: "blue", emoji: "\u{1F48E}" }
      : { label: "Lucky", color: "green", emoji: "\u{1F340}" };
    return { ...tier, text: `${streak} in a row — ${tier.label}`, key: Math.floor(streak / 5) };
  }

  // Colour class of the big percentage: zone beats sign.
  function biasTone(biasPct, inZone) {
    return inZone ? "zone" : biasPct > 0 ? "pos" : biasPct < 0 ? "neg" : "zero";
  }

  function fmtSignedPct(x, digits) {
    return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}%`;
  }
  function fmtThreshold(t) { return `${t.toFixed(1)}%`; }
  function fmtSigma(z) {
    if (typeof z !== "number" || !Number.isFinite(z)) throw new TypeError(`sigma must be a finite number, got ${z}`);
    return `${z.toFixed(2)}σ`;
  }

  // Qubit count rules: majority needs an odd count (the server also validates this).
  function qubitRange(aggregation) {
    if (aggregation === "none") return { min: 1, max: 1, step: 1 };
    if (aggregation === "xor") return { min: 1, max: 16, step: 1 };
    if (aggregation === "majority") return { min: 1, max: 15, step: 2 };
    throw new RangeError(`unknown aggregation ${aggregation}`);
  }
  // Nearest valid count for the aggregation (ties go down).
  function normalizeQubits(n, aggregation) {
    const r = qubitRange(aggregation);
    let v = Math.max(r.min, Math.min(r.max, Math.round(n)));
    if (aggregation === "majority" && v % 2 === 0) v = v - 1 >= r.min ? v - 1 : v + 1;
    return v;
  }
  function stepQubits(n, delta, aggregation) {
    const r = qubitRange(aggregation);
    return normalizeQubits(Math.max(r.min, Math.min(r.max, n + delta * r.step)), aggregation);
  }

  // Unconsumed shots of the pool(s) a buffered session would draw from. A pool key is
  // provider__implementation__qubits; the ibm implementation is unknown until first use, so
  // without one every pool of the provider with that qubit count matches.
  function poolAvailable(pools, provider, implementation, n) {
    const matching = pools.filter((p) => p.provider === provider && p.n_qubits === n &&
      (!implementation || p.implementation === implementation));
    return { shots: matching.reduce((a, p) => a + p.available_shots, 0), pools: matching };
  }

  // The Pool line text. With no implementation (ibm before its device is resolved) pools of
  // devices must not be summed into one "ready" number, and even a single listed device is not
  // "ready": least_busy may pick another device (empty pool) at the first job.
  function poolLine(pools, provider, implementation, n, fmt) {
    const found = poolAvailable(pools, provider, implementation, n);
    const where = `${provider}${implementation ? "/" + implementation : ""}, ${n} qubit${n === 1 ? "" : "s"}`;
    if (found.shots <= 0) return `Pool ${where}: empty. Prefetch before starting.`;
    if (!implementation) {
      const per = found.pools.map((p) => `${p.implementation} ${fmt(p.available_shots)}`).join(", ");
      return `Pool ${where}: ${per} shots; the device is chosen at the first job`;
    }
    return `Pool ${where}: ${fmt(found.shots)} shots ready`;
  }

  // ---------------------------------------------------------------- session state reducers
  // Every response or event is applied to the focus state `F` only if it belongs to the session
  // that is shown now (`sid`). A request that outlives its session (exit, then a new Start) must
  // not write into the next one. Each reducer returns whether it applied.
  function owns(F, sid) { return F.session !== null && F.session !== undefined && F.session.id === sid; }

  function applyStats(F, sid, stats) {
    if (!owns(F, sid)) return false;
    // Responses and SSE events can arrive out of order; counts only grow.
    if (stats.total.shots < F.statsShots) return false;
    F.statsShots = stats.total.shots;
    const a = stats.total.aggregated;
    F.bias = a.bias_pct; F.z = a.z; F.n = a.n;
    if (F.kind === "predict" || F.kind === "perceive") {
      const g = stats.total[F.kind];
      if (!g) throw new Error(`stats.total.${F.kind} missing from the server response`);
      F.guessStats = g;
      F.streak = g.streak;
    }
    return true;
  }
  function applyPress(F, sid, r) {
    if (!owns(F, sid)) return false;
    F.streak = nextStreak(F.streak, r.bit, F.target);
    F.prov = r.provenance;
    applyStats(F, sid, r.stats);
    return true;
  }
  function applyGuess(F, sid, r) {
    if (!owns(F, sid)) return false;
    F.result = { bit: r.bit, guess: r.guess, hit: r.hit, before: r.measured_before_guess === true };
    F.prov = r.provenance;
    applyStats(F, sid, r.stats);
    return true;
  }
  function applyPrepare(F, sid, trial) {
    if (!owns(F, sid)) return false;
    F.result = null;
    F.trial = trial;
    return true;
  }
  function applyReveal(F, sid, r) {
    if (!owns(F, sid)) return false;
    F.trial = null;
    return applyGuess(F, sid, r);
  }
  function applyBatch(F, sid, d) {
    if (!owns(F, sid)) return false;
    if (F.kind === "stream") F.streak = batchStreak(F.streak, d.tail_bit, d.tail_run, d.shots, F.target);
    F.prov = d.provenance;
    return true;
  }
  // The session's `stream` object (response, session read or stream event). The server sends
  // seconds_left; the deadline is kept as a local time so the countdown ticks without polling.
  function applyStream(F, sid, stream, nowMs) {
    if (!owns(F, sid)) return false;
    F.streaming = stream.running;
    F.streamInfo = stream;
    F.deadlineAt = stream.running && stream.seconds_left !== null && stream.seconds_left !== undefined
      ? nowMs + stream.seconds_left * 1000 : null;
    return true;
  }

  // Status line of the stream bar: the countdown while running, the limit message once a limit
  // stopped it, null otherwise. Jobs are shown wherever a limit counts them: ibm (live cycles and stream-triggered refills).
  function streamBar(F, nowMs) {
    const st = F.streamInfo;
    if (!st || !F.session) return null;
    if (F.streaming) {
      if (F.deadlineAt === null || F.deadlineAt === undefined) throw new Error("running stream without seconds_left");
      const secs = Math.max(0, Math.ceil((F.deadlineAt - nowMs) / 1000));
      const jobs = F.session.provider === "ibm" && st.jobs_limit !== null
        ? ` \u00b7 jobs ${st.jobs_used}/${st.jobs_limit}` : "";
      return { text: `Stream: ${secs} s left${jobs}`, tone: "run" };
    }
    if (st.stopped_reason) {
      if (st.stopped_reason.startsWith("limit_seconds")) return { text: `Stream stopped: ${st.seconds_limit} s limit reached`, tone: "limit" };
      if (st.stopped_reason.startsWith("limit_jobs")) return { text: `Stream stopped: ${st.jobs_limit} job limit reached`, tone: "limit" };
      throw new Error(`unknown stream stop reason ${st.stopped_reason}`);
    }
    return null;
  }

  // Setup fields to the create-body fields. A positive integer or an Error naming the field; the
  // server stays the authority (it also bounds them).
  function positiveInt(text, label) {
    if (!/^[0-9]+$/.test(String(text).trim()) || Number(text) < 1) {
      throw new Error(`${label} must be a whole number of at least 1, got "${text}"`);
    }
    return Number(text);
  }
  function streamLimits(secondsText, jobsText, ibm) {
    const out = { stream_seconds: positiveInt(secondsText, "Stream duration") };
    if (ibm) out.stream_max_jobs = positiveInt(jobsText, "Max IBM jobs");
    return out;
  }
  // Worst-case QPU seconds of a stream: max jobs times the estimate for one job.
  function streamQpuEstimate(est, shotsPerJob, maxJobs) {
    return estimateQpuSeconds(est, shotsPerJob) * maxJobs;
  }

  // ---------------------------------------------------------------- job state line
  // "job <id> <state>, queue position N | not reported, elapsed 12s". `nowMs` is passed in.
  function jobStatusText(job, nowMs) {
    const submitted = Date.parse(job.submitted_at);
    if (!Number.isFinite(submitted)) throw new Error(`job ${job.id} has no valid submitted_at: ${job.submitted_at}`);
    const end = job.finished_at ? Date.parse(job.finished_at) : nowMs;
    if (!Number.isFinite(end)) throw new Error(`job ${job.id} has no valid finished_at: ${job.finished_at}`);
    const secs = Math.max(0, Math.floor((end - submitted) / 1000));
    const q = job.queue_position === null || job.queue_position === undefined ? "not reported" : String(job.queue_position);
    return `job ${job.job_id || job.id} ${job.state}, queue position ${q}, elapsed ${secs}s`;
  }

  // Estimated QPU seconds of a job of `shots` shots from the server's /api/budget `qpu_estimate`.
  function estimateQpuSeconds(est, shots) {
    if (!est || !Number.isFinite(est.base_seconds) || !Number.isFinite(est.seconds_per_shot)) {
      throw new Error("budget response has no qpu_estimate");
    }
    return est.base_seconds + est.seconds_per_shot * shots;
  }

  const api = {
    applyStats, applyPress, applyGuess, applyPrepare, applyReveal, applyBatch, applyStream,
    jobStatusText, estimateQpuSeconds, streamBar, streamLimits, streamQpuEstimate,
    DEFAULT_THRESHOLD_PCT, FIRE_THRESHOLD, SPRING,
    ballPosition, inTargetZone, springStep, springSettled, nextStreak, batchStreak,
    fireTier, streakLabel, biasTone, fmtSignedPct, fmtThreshold, fmtSigma,
    qubitRange, normalizeQubits, stepQubits, poolAvailable, poolLine,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.QL = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
