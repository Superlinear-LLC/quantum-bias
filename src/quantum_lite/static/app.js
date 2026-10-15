"use strict";
/* Quantum Bias Lab UI v2: a setup page and a full-screen focus view. Vanilla JS, no build step,
   no external assets. Server text is only ever written with textContent. Errors are shown
   verbatim (code: detail), never replaced by a default. */

const $ = (id) => document.getElementById(id);
const FINAL_STATES = ["done", "error", "cancelled"];
const THRESHOLD = QL.DEFAULT_THRESHOLD_PCT;
const FIRE_THRESHOLD = QL.FIRE_THRESHOLD;

// ---------------------------------------------------------------- state
const S = {
  providers: [],
  pools: [],
  budget: null,
  goal: "affect",
  sub: { affect: "key_press", guess: "predict" },
  target: 1,
  source: "buffered",
  agg: "none",
  qubits: 1,
  errors: [], // [{msg}] newest last, at most 3
  seenErrors: new Set(),
  globalSse: null,
  sessionSse: null,
  starting: false,
};

// What the focus view shows. `session` is null only in the test hook.
const F = {
  session: null,
  kind: "key_press", // key_press | stream | predict | perceive
  target: 1,
  bias: 0, z: 0, n: 0, // aggregated bias % and sigma of the session
  streak: 0, // Affect: consecutive target bits. Guess: hits in a row (from the server).
  streaming: false,
  trial: null, // perceive: {trial_id, source}
  result: null, // last guess result
  guessStats: null, // stats.total.predict | perceive
  statsShots: 0,
  prov: null, // provenance of the latest bit/batch
  job: null, // latest job of this session
  busy: 0,
  streamErrorShown: null, // last stream error already on screen (event or session read)
  streamInfo: null, // session.stream: limits, jobs used, stopped_reason
  deadlineAt: null, // Date.now() at which the running stream's duration ends
  hook: false,
};

// ---------------------------------------------------------------- helpers
function el(tag, props, ...kids) {
  const e = document.createElement(tag);
  if (props) for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}
function fmtInt(x) { return Number(x).toLocaleString("en-US"); }

class ApiError extends Error {
  constructor(status, code, detail) {
    super(`${code}: ${detail}`);
    this.status = status; this.code = code; this.detail = detail;
  }
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const resp = await fetch(path, opts);
  const text = await resp.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); }
    catch (_) { throw new ApiError(resp.status, "non_json_response", `${method} ${path} -> HTTP ${resp.status}: ${text.slice(0, 500)}`); }
  }
  if (!resp.ok) {
    if (data && typeof data === "object" && "error" in data) throw new ApiError(resp.status, data.error, data.detail);
    throw new ApiError(resp.status, "http_error", `${method} ${path} -> HTTP ${resp.status}: ${text.slice(0, 500)}`);
  }
  return data;
}

// ---------------------------------------------------------------- errors (destructive alert)
function showError(msg, sid) {
  msg = String(msg);
  if (sid === undefined) sid = F.session ? F.session.id : null; // session-scoped errors die with their session
  if (S.errors.length && S.errors[S.errors.length - 1].msg === msg) return;
  S.errors.push({ msg, sid });
  if (S.errors.length > 3) S.errors.shift();
  renderErrors();
}
function dismissError(i) { S.errors.splice(i, 1); renderErrors(); }
function renderErrors() {
  for (const id of ["alerts-setup", "alerts-focus"]) {
    $(id).replaceChildren(...S.errors.map((e, i) => {
      const pre = el("pre"); pre.textContent = e.msg;
      const x = el("button", { class: "dismiss", type: "button", "aria-label": "Dismiss error" }); x.textContent = "×";
      x.addEventListener("click", () => dismissError(i));
      const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      icon.setAttribute("class", "icon");
      const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
      use.setAttribute("href", "#i-alert");
      icon.append(use);
      return el("div", { class: "alert", role: "alert" }, icon, pre, x);
    }));
  }
}
window.addEventListener("error", (e) => showError(`JS error: ${e.message} (${e.filename}:${e.lineno})`));
window.addEventListener("unhandledrejection", (e) => showError(e.reason instanceof Error ? e.reason.message : `Unhandled rejection: ${e.reason}`));

/** Run an async action; show whatever it throws, verbatim. */
async function guarded(fn) {
  try { return await fn(); }
  catch (err) { showError(err instanceof Error ? err.message : String(err)); return undefined; }
}

// ================================================================ SETUP
function pressed(node, on) { node.setAttribute("aria-pressed", String(on)); }

function renderSetup() {
  for (const b of document.querySelectorAll("[data-goal]")) pressed(b, b.dataset.goal === S.goal);
  $("sub-affect").hidden = S.goal !== "affect";
  $("sub-guess").hidden = S.goal !== "guess";
  for (const b of document.querySelectorAll("[data-sub]")) {
    const group = b.closest("#sub-affect") ? "affect" : "guess";
    pressed(b, S.sub[group] === b.dataset.sub);
  }
  $("group-target").hidden = S.goal !== "affect";
  for (const b of document.querySelectorAll("[data-target]")) pressed(b, Number(b.dataset.target) === S.target);
  $("target-note").textContent = `In the focus view, the ball moves toward your target (${S.target}).`;
  for (const b of document.querySelectorAll("[data-source]")) pressed(b, b.dataset.source === S.source);
  for (const b of document.querySelectorAll("[data-agg]")) pressed(b, b.dataset.agg === S.agg);
  const q = $("qubits");
  const r = QL.qubitRange(S.agg);
  $("qubits-row").hidden = S.agg === "none";
  q.min = r.min; q.max = r.max; q.step = r.step; q.value = S.qubits;
  $("q-minus").disabled = S.qubits <= r.min;
  $("q-plus").disabled = S.qubits + r.step > r.max;
  $("qubits-note").textContent = S.agg === "majority"
    ? "Majority needs an odd count." : `${S.qubits} qubit${S.qubits === 1 ? "" : "s"} per shot`;
  renderPool();
  renderStreamLimits();
}

function setQubits(n) {
  S.qubits = QL.normalizeQubits(n, S.agg);
  renderSetup();
}

function currentProvider() { return S.providers.find((p) => p.id === $("provider").value); }

function renderProviders() {
  const sel = $("provider");
  const keep = sel.value;
  sel.replaceChildren(...S.providers.map((p) => {
    const o = el("option", { value: p.id });
    const impl = p.implementation ? `, ${p.implementation}` : "";
    o.textContent = `${p.id} (${p.variant}${impl})` + (p.available ? "" : ` — unavailable: ${p.reason}`);
    o.disabled = !p.available;
    return o;
  }));
  const ok = S.providers.find((p) => p.available && p.id === keep) || S.providers.find((p) => p.available);
  if (ok) sel.value = ok.id;
  renderProviderNote();
}

function renderProviderNote() {
  const p = currentProvider();
  const unavailable = S.providers.filter((x) => !x.available);
  const lines = unavailable.map((x) => `${x.id} unavailable: ${x.reason}`);
  if (p && p.id === "ibm") lines.push("ibm runs on a real QPU: every job is real and counts against the budget in Details.");
  $("provider-note").textContent = lines.join(" ");
}

function renderPool() {
  const line = $("pool-line");
  line.hidden = S.source !== "buffered";
  if (S.source !== "buffered") return;
  const p = currentProvider();
  if (!p) { $("pool-text").textContent = "No available provider."; return; }
  $("pool-text").textContent = QL.poolLine(S.pools, p.id, p.implementation, S.qubits, fmtInt);
}

async function loadProviders() {
  S.providers = await api("GET", "/api/providers");
  renderProviders();
  renderPool();
  renderStreamLimits();
}
async function loadPools() { S.pools = await api("GET", "/api/pools"); renderPool(); }
async function loadBudget() {
  S.budget = await api("GET", "/api/budget");
  const b = S.budget;
  $("budget-line").textContent = `IBM budget: ${fmtInt(b.shots.used)}/${fmtInt(b.shots.ceiling)} shots, ` +
    `${b.jobs.used}/${b.jobs.ceiling} jobs, max ${fmtInt(b.max_shots_per_job)} shots per job`;
  renderPrefetchEstimate();
  renderStreamLimits();
}
// Stream limits are asked for when Stream is the chosen interaction; Max IBM jobs only on ibm.
function renderStreamLimits() {
  const streamKind = S.goal === "affect" && S.sub.affect === "stream";
  $("group-stream").hidden = !streamKind;
  const p = currentProvider();
  const ibm = !!p && p.id === "ibm";
  $("stream-jobs-row").hidden = !ibm;
  const node = $("stream-estimate");
  node.textContent = "";
  if (!streamKind || !ibm || !S.budget) return;
  const jobs = $("stream-max-jobs").value.trim(), shots = $("stream-shots").value.trim();
  if (!/^[0-9]+$/.test(jobs) || !/^[0-9]+$/.test(shots) || Number(jobs) < 1) return; // Start Training reports it
  const secs = QL.streamQpuEstimate(S.budget.qpu_estimate, Number(shots), Number(jobs));
  node.textContent = S.source === "live"
    ? `At most ${jobs} job${jobs === "1" ? "" : "s"} of ${shots} shots: estimated QPU time up to ${secs.toFixed(1)} s ` +
      `(${S.budget.qpu_estimate.formula} per job, an estimate)`
    : `Buffered streams submit no jobs. If the session goes live: up to ${secs.toFixed(1)} s of QPU time (an estimate).`;
}

function renderPrefetchEstimate() {
  const p = currentProvider();
  const node = $("prefetch-estimate");
  if (!p || p.id !== "ibm") { node.textContent = ""; return; }
  const secs = QL.estimateQpuSeconds(S.budget.qpu_estimate, Number($("prefetch-shots").value));
  node.textContent = `estimated QPU time ${secs.toFixed(1)} s (${S.budget.qpu_estimate.formula}, an estimate)`;
}

function wireSetup() {
  for (const b of document.querySelectorAll("[data-goal]")) b.addEventListener("click", () => { S.goal = b.dataset.goal; renderSetup(); });
  for (const b of document.querySelectorAll("[data-sub]")) b.addEventListener("click", () => {
    S.sub[b.closest("#sub-affect") ? "affect" : "guess"] = b.dataset.sub; renderSetup();
  });
  for (const b of document.querySelectorAll("[data-target]")) b.addEventListener("click", () => { S.target = Number(b.dataset.target); renderSetup(); });
  for (const b of document.querySelectorAll("[data-source]")) b.addEventListener("click", () => { S.source = b.dataset.source; renderSetup(); });
  for (const b of document.querySelectorAll("[data-agg]")) b.addEventListener("click", () => {
    S.agg = b.dataset.agg;
    S.qubits = S.agg === "none" ? 1 : QL.normalizeQubits(S.qubits === 1 ? 3 : S.qubits, S.agg);
    renderSetup();
  });
  $("q-minus").addEventListener("click", () => setQubits(QL.stepQubits(S.qubits, -1, S.agg)));
  $("q-plus").addEventListener("click", () => setQubits(QL.stepQubits(S.qubits, 1, S.agg)));
  $("qubits").addEventListener("change", () => setQubits(Number($("qubits").value)));
  $("provider").addEventListener("change", () => {
    renderProviderNote(); renderPool(); renderStreamLimits();
    $("prefetch-shots").value = $("provider").value === "ibm" ? 20000 : 5000;
    guarded(async () => renderPrefetchEstimate());
  });
  for (const id of ["stream-max-jobs", "stream-shots"]) $(id).addEventListener("input", () => guarded(async () => renderStreamLimits()));
  $("prefetch-shots").addEventListener("input", () => guarded(async () => renderPrefetchEstimate()));
  $("btn-prefetch").addEventListener("click", () => guarded(prefetch));
  $("btn-start").addEventListener("click", () => guarded(startTraining));
}

async function prefetch() {
  const p = currentProvider();
  if (!p) throw new Error("no available provider to prefetch from");
  const r = await api("POST", "/api/pools/prefetch", {
    provider: p.id, num_qubits: S.qubits, shots: Number($("prefetch-shots").value),
  });
  onJob(r.job);
}

// ================================================================ SESSION
async function startTraining() {
  if (S.starting) return;
  S.starting = true;
  const btn = $("btn-start");
  btn.disabled = true; btn.textContent = "Creating Session...";
  try {
    const kind = S.goal === "affect" ? S.sub.affect : S.sub.guess;
    const body = {
      provider: $("provider").value,
      mode: S.source,
      num_qubits: S.qubits,
      aggregation: S.agg,
      shots_per_job: Number($("stream-shots").value),
      interval_ms: Number($("stream-interval").value),
      auto_refill: false,
      refill_below: 0,
      ...QL.streamLimits($("stream-seconds").value, $("stream-max-jobs").value, $("provider").value === "ibm"),
    };
    const session = await api("POST", "/api/sessions", body);
    openFocus(session, kind);
    guarded(loadProviders); // starting resolved the ibm device: implementation is no longer null
  } finally {
    S.starting = false;
    btn.disabled = false; btn.textContent = "Start Training";
  }
}

function openFocus(session, kind) {
  closeSessionSse();
  S.errors = S.errors.filter((e) => e.sid === null);
  renderErrors();
  Object.assign(F, {
    session, kind, target: S.target, bias: 0, z: 0, n: 0, streak: 0, streaming: false, trial: null,
    result: null, guessStats: null, statsShots: 0, prov: null, job: null, busy: 0, hook: false,
    streamErrorShown: null, streamInfo: session.stream, deadlineAt: null,
  });
  $("focus").classList.remove("busy"); // a slow request of the previous session may have dimmed it
  applyStats(session.stats);
  $("setup").hidden = true;
  $("focus").hidden = false;
  document.activeElement && document.activeElement.blur();
  openSessionSse(session.id);
  buildFocus();
  renderFocus(true);
}

async function exitFocus() {
  if ($("focus").hidden) return;
  const s = F.session;
  try {
    if (s && F.streaming) {
      const resp = await api("POST", `/api/sessions/${s.id}/stream/stop`);
      F.streaming = resp.stream.running;
    }
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  }
  closeSessionSse(); // the server also stops a stream whose last listener is gone
  F.session = null; F.streaming = false; F.job = null;
  $("focus").classList.remove("busy");
  stopJobTimer();
  stopStreamTimer();
  stopBallLoop();
  $("focus").hidden = true;
  $("setup").hidden = false;
  guarded(loadPools); guarded(loadBudget);
}

function applyStats(stats) { return QL.applyStats(F, F.session && F.session.id, stats); }

// ---------------------------------------------------------------- SSE
function parse(ev) {
  const d = JSON.parse(ev.data);
  if (d.dropped) showError(`Event feed dropped ${d.dropped} event(s) (this browser reads too slowly); shown state may be stale.`);
  return d;
}

function renderConn() {
  const g = S.globalSse && S.globalSse.readyState === EventSource.OPEN;
  const s = S.sessionSse ? S.sessionSse.readyState === EventSource.OPEN : null;
  $("conn-line").textContent = `Events: ${g ? "live" : "not connected"}` + (s === null ? "" : `, session feed ${s ? "live" : "not connected"}`);
}

let connLost = false;
function onFeedError(name, es) {
  renderConn();
  if (!connLost) {
    connLost = true;
    showError(`Event feed "${name}" error (EventSource readyState=${es.readyState}). The browser will retry.`);
  }
}
function onFeedOpen() { connLost = false; renderConn(); }

function openGlobalSse() {
  const es = new EventSource("/api/events");
  S.globalSse = es;
  es.onopen = onFeedOpen;
  es.onerror = (e) => { if (e.data === undefined) onFeedError("global", es); };
  es.addEventListener("job", (e) => guarded(() => onJob(parse(e))));
  es.addEventListener("pool", (e) => guarded(() => onPool(parse(e))));
  es.addEventListener("heartbeat", (e) => guarded(() => parse(e)));
  es.addEventListener("error", (e) => { if (e.data) guarded(() => onErrorEvent(parse(e))); });
}

function openSessionSse(id) {
  const es = new EventSource(`/api/sessions/${id}/events`);
  S.sessionSse = es;
  // On every (re)connect re-read the session: the server may have stopped its stream while the
  // feed was down (last listener gone), and no event carries that.
  es.onopen = () => { onFeedOpen(); guarded(refreshSession); };
  es.onerror = (e) => { if (e.data === undefined) onFeedError("session", es); };
  es.addEventListener("stats", (e) => guarded(() => onStats(parse(e))));
  es.addEventListener("batch", (e) => guarded(() => onBatch(parse(e))));
  es.addEventListener("job", (e) => guarded(() => onJob(parse(e))));
  es.addEventListener("stream", (e) => guarded(() => onStream(parse(e))));
  es.addEventListener("stream_stopped", (e) => guarded(() => onStream(parse(e))));
  es.addEventListener("bit", (e) => guarded(() => parse(e))); // results come from the HTTP responses that caused them
  es.addEventListener("heartbeat", (e) => guarded(() => parse(e)));
  es.addEventListener("error", (e) => { if (e.data) guarded(() => onErrorEvent(parse(e))); });
}

function closeSessionSse() {
  if (S.sessionSse) { S.sessionSse.close(); S.sessionSse = null; }
  renderConn();
}

async function refreshSession() {
  const s = F.session;
  if (!s) return;
  const fresh = await api("GET", `/api/sessions/${s.id}`);
  if (F.session !== s) return;
  QL.applyStream(F, s.id, fresh.stream, Date.now());
  // A stream failure is also pushed as an error event, which onErrorEvent already showed.
  if (fresh.stream.error && fresh.stream.error !== F.streamErrorShown) showError(fresh.stream.error);
  F.streamErrorShown = fresh.stream.error;
  renderFocus();
}

function onErrorEvent(d) {
  const key = `${d.ts}|${d.scope}|${d.error}|${d.detail}`;
  if (S.seenErrors.has(key)) return;
  S.seenErrors.add(key);
  showError(`${d.scope || "error"} / ${d.error}: ${d.detail}` + (d.pool_key ? `  [pool ${d.pool_key}]` : ""));
  if (d.scope === "stream" && F.session && d.session_id === F.session.id) {
    F.streamErrorShown = `${d.error}: ${d.detail}`; // the form session.stream.error has
    guarded(refreshSession);
  }
}

function onStats(d) {
  if (!QL.applyStats(F, d.session_id, d.stats)) return;
  renderFocus();
}

// A stream state pushed by the server: a run started, a live job was submitted, or a limit
// stopped the run (stopped_reason set; a limit is not an error and shows in the stream bar).
function onStream(d) {
  if (!QL.applyStream(F, d.session_id, d.stream, Date.now())) return;
  renderFocus();
}

function onBatch(d) {
  if (!QL.applyBatch(F, d.session_id, d)) return;
  renderFocus();
}

function onJob(j) {
  $("job-line").textContent = `Latest job: ${j.job_id || j.id} ${j.provider}/${j.backend || "?"} ${j.kind} ` +
    `${j.shots} shots, ${j.state}${j.error ? " — " + j.error : ""}`;
  if (j.kind === "prefetch") {
    if (j.provider === "ibm" && FINAL_STATES.includes(j.state)) guarded(loadProviders);
    $("pool-job").textContent = `Prefetch job ${j.job_id || j.id}: ${j.state}` +
      (j.queue_position != null ? ` (queue position ${j.queue_position})` : "") + (j.error ? ` — ${j.error}` : "");
  }
  if (F.session && j.session_id === F.session.id && j.kind !== "prefetch") { F.job = j; renderProvenance(); }
  if (j.provider === "ibm") guarded(loadBudget);
}

function onPool(d) {
  const i = S.pools.findIndex((p) => p.key === d.pool.key);
  if (i >= 0) S.pools[i] = d.pool; else S.pools.push(d.pool);
  renderPool();
}

// ================================================================ ACTIONS
async function withBusy(fn) {
  const s = F.session;
  if (F.busy || !s) return;
  F.busy = 1;
  const t = setTimeout(() => { if (F.session === s) $("focus").classList.add("busy"); }, 250);
  try { return await fn(s.id); }
  catch (err) { showError(err instanceof Error ? err.message : String(err), s.id); return undefined; }
  finally {
    clearTimeout(t);
    // A request that outlived its session must not touch the busy state of the next one.
    if (F.session === s) { F.busy = 0; $("focus").classList.remove("busy"); renderFocus(); }
  }
}

function doPrimary() {
  if (!F.session) return;
  if (F.kind === "key_press") return withBusy(press);
  if (F.kind === "stream") return toggleStream();
  if (F.kind === "perceive" && !F.trial) return withBusy(prepare);
}

async function press(sid) {
  const r = await api("POST", `/api/sessions/${sid}/press`);
  QL.applyPress(F, sid, r);
}

async function toggleStream() {
  return withBusy(async (sid) => {
    const r = await api("POST", `/api/sessions/${sid}/stream/${F.streaming ? "stop" : "start"}`);
    QL.applyStream(F, sid, r.stream, Date.now());
  });
}

async function prepare(sid) {
  QL.applyPrepare(F, sid, null);
  QL.applyPrepare(F, sid, await api("POST", `/api/sessions/${sid}/perceive/prepare`));
}

function doGuess(g) {
  if (!F.session) return;
  if (F.kind === "predict") return withBusy(async (sid) => {
    QL.applyGuess(F, sid, await api("POST", `/api/sessions/${sid}/predict`, { guess: g }));
  });
  if (F.kind === "perceive" && F.trial) return withBusy(async (sid) => {
    const trialId = F.trial.trial_id;
    QL.applyReveal(F, sid, await api("POST", `/api/sessions/${sid}/perceive/reveal`, { trial_id: trialId, guess: g }));
  });
}

// ================================================================ FOCUS VIEW
const HINTS = {
  key_press: [["Tap/click the card or press ", { k: "SPACE" }, " to request a bit"], ["Press ", { k: "F" }, " to exit focus mode"]],
  stream: [["Tap/click the card or press ", { k: "S" }, " to start or stop the stream (or use Stop stream)"], ["Press ", { k: "F" }, " to exit focus mode"]],
  predict: [["Press ", { k: "0" }, " or ", { k: "1" }, " to guess the next bit"], ["Press ", { k: "F" }, " to exit focus mode"]],
  perceive: [["Press ", { k: "SPACE" }, " to request a hidden bit, then ", { k: "0" }, " or ", { k: "1" }, " to guess"], ["Press ", { k: "F" }, " to exit focus mode"]],
};

function buildFocus() {
  const affect = F.kind === "key_press" || F.kind === "stream";
  $("ball-card").hidden = !affect;
  $("guess-card").hidden = affect;
  $("focus-title").textContent = affect ? `Target: ${F.target}` : (F.kind === "predict" ? "Predict" : "Perceive");
  $("ball-card").setAttribute("aria-label", F.kind === "stream" ? "Start or stop the stream" : "Request quantum bit");
  $("hints").replaceChildren(...HINTS[F.kind].map((parts) => {
    const d = el("div");
    for (const p of parts) {
      if (typeof p === "string") d.append(p); else { const k = el("kbd"); k.textContent = p.k; d.append(k); }
    }
    return d;
  }));
  if (affect) {
    $("track").className = `track t${F.target}`;
    const zone = $("track-zone");
    zone.className = `track-zone ${F.target === 0 ? "left" : "right"}`;
    const L = $("lbl-left"), C = $("lbl-center"), R = $("lbl-right");
    L.textContent = `-${QL.fmtThreshold(THRESHOLD)}`;
    L.className = F.target === 0 ? "green bold" : "red";
    C.textContent = `Target: ${F.target === 0 ? "Zero" : "One"}`;
    R.textContent = `+${QL.fmtThreshold(THRESHOLD)}`;
    R.className = F.target === 1 ? "green bold" : "green";
  }
}

function renderFocus(initial) {
  if ($("focus").hidden) return;
  if (F.kind === "key_press" || F.kind === "stream") renderViz(initial);
  else renderGuess();
  renderStreamBar();
  renderProvenance();
}

// ---------------------------------------------------------------- stream bar: countdown, Stop, limit message
function renderStreamBar() {
  const bar = $("stream-bar");
  const info = F.kind === "stream" && !F.hook ? QL.streamBar(F, Date.now()) : null;
  bar.hidden = info === null;
  $("btn-stream-stop").hidden = !F.streaming;
  if (info === null) { stopStreamTimer(); return; }
  const st = $("stream-status");
  st.textContent = info.text;
  st.className = `mono stream-status ${info.tone}`;
  if (F.streaming && !streamTimer) streamTimer = setInterval(() => guarded(async () => renderStreamBar()), 250);
  else if (!F.streaming) stopStreamTimer();
}
let streamTimer = 0;
function stopStreamTimer() { if (streamTimer) { clearInterval(streamTimer); streamTimer = 0; } }

function streakMsg(container, streak) {
  const m = QL.streakLabel(streak, FIRE_THRESHOLD);
  if (!m) { container.replaceChildren(); container.dataset.key = ""; return; }
  if (container.dataset.key === String(m.key) && container.firstChild) {
    // same tier bucket: update the text without re-running the pop animation
    container.firstChild.querySelector(".text").textContent = m.text;
    container.firstChild.className = `msg ${m.color}`;
    return;
  }
  container.dataset.key = String(m.key);
  const wrap = el("span", { class: `msg ${m.color}` });
  const a = el("span", { class: "emoji" }); a.textContent = m.emoji;
  const t = el("span", { class: "text" }); t.textContent = m.text;
  const b = el("span", { class: "emoji" }); b.textContent = m.emoji;
  wrap.append(a, t, b);
  container.replaceChildren(wrap);
}

// ---------------------------------------------------------------- ball
const BALL = {
  spring: { x: 50, v: 0 }, target: 50, raf: 0, last: 0, rot: 0, tier: -1, flameTimer: 0, init: false,
};
const GRADIENTS = {
  target: "radial-gradient(circle at 30% 30%, #fef08a 0%, #facc15 50%, #ca8a04 100%)",
  green: "radial-gradient(circle at 30% 30%, #86efac 0%, #22c55e 50%, #15803d 100%)",
  red: "radial-gradient(circle at 30% 30%, #fca5a5 0%, #ef4444 50%, #b91c1c 100%)",
  fire: "radial-gradient(circle at 30% 30%, #fff7ed 0%, #fb923c 40%, #ea580c 70%, #9a3412 100%)",
};
const FIRE_PARTICLES = [
  { x: -26, y: -2, size: 7, drift: -20, rise: 34, duration: 0.95, delay: 0.0 },
  { x: -18, y: 8, size: 5, drift: -12, rise: 28, duration: 0.8, delay: 0.18 },
  { x: -8, y: 2, size: 6, drift: -8, rise: 30, duration: 0.9, delay: 0.08 },
  { x: 0, y: 10, size: 4, drift: 0, rise: 24, duration: 0.75, delay: 0.22 },
  { x: 10, y: 6, size: 5, drift: 8, rise: 28, duration: 0.85, delay: 0.32 },
  { x: 18, y: 2, size: 6, drift: 14, rise: 32, duration: 0.95, delay: 0.12 },
  { x: 25, y: 7, size: 4, drift: 18, rise: 24, duration: 0.7, delay: 0.26 },
  { x: 4, y: -4, size: 5, drift: 4, rise: 26, duration: 0.82, delay: 0.4 },
];
// Flame frames (fire tier 4): ignition once, then a 3-frame sustain loop.
const IGNITION_FRAMES = [
  { outer: "M50 102 C47 99 46 95 47 91 C48 87 49 84 50 80 C51 84 52 87 53 91 C54 95 53 99 50 102 Z", inner: "M50 99 C48.5 96.5 48 93.7 49 91 C49.5 88.7 49.8 86.8 50 85 C50.2 86.8 50.5 88.7 51 91 C52 93.7 51.5 96.5 50 99 Z", scale: 0.7, offsetY: 14, opacity: 0.7 },
  { outer: "M50 102 C46 97 44 91 45 85 C46 79 48 73 50 66 C52 73 54 79 55 85 C56 91 54 97 50 102 Z", inner: "M50 99 C47.7 95.5 47 91.8 47.8 87.8 C48.3 84.8 49.1 81.8 50 78 C50.9 81.8 51.7 84.8 52.2 87.8 C53 91.8 52.3 95.5 50 99 Z", scale: 0.8, offsetY: 10, opacity: 0.78 },
  { outer: "M50 102 C43 95 40 87 41 78 C42.5 67 46.5 58 50 47 C53.5 58 57.5 67 59 78 C60 87 57 95 50 102 Z", inner: "M50 98 C46.5 92.5 45.4 86.4 46.3 79.8 C47.1 73.3 48.7 66.7 50 60 C51.3 66.7 52.9 73.3 53.7 79.8 C54.6 86.4 53.5 92.5 50 98 Z", scale: 0.9, offsetY: 7, opacity: 0.86 },
  { outer: "M50 102 C40 94 35 84 36.5 70 C38 57 44 47 50 32 C56 47 62 57 63.5 70 C65 84 60 94 50 102 Z", inner: "M50 97 C44.4 90.2 42.8 82.8 43.8 74.5 C45 65.8 47.8 56.5 50 47 C52.2 56.5 55 65.8 56.2 74.5 C57.2 82.8 55.6 90.2 50 97 Z", scale: 0.98, offsetY: 2, opacity: 0.94 },
  { outer: "M50 102 C38 94 33 82 35 68 C36.8 54.8 43.2 43.8 50 24 C56.8 43.8 63.2 54.8 65 68 C67 82 62 94 50 102 Z", inner: "M50 96 C43.2 89.2 41.2 80.8 42.6 70.8 C44.1 61 47.6 49.5 50 39 C52.4 49.5 55.9 61 57.4 70.8 C58.8 80.8 56.8 89.2 50 96 Z", scale: 1.03, offsetY: -1, opacity: 1 },
];
const SUSTAIN_FRAMES = [
  { outer: "M50 102 C38 95 33 84 35 69 C36.7 56.5 42.6 46.3 50 24 C57.4 46.3 63.3 56.5 65 69 C67 84 62 95 50 102 Z", inner: "M50 96 C43.1 89.6 41.2 81.4 42.6 71.3 C44 62.5 47.3 51.2 50 40 C52.7 51.2 56 62.5 57.4 71.3 C58.8 81.4 56.9 89.6 50 96 Z", scale: 1.03, offsetY: -2, opacity: 1 },
  { outer: "M50 102 C37 94 32 83 34 68 C35.9 55.1 42 44.4 50 22 C58 44.4 64.1 55.1 66 68 C68 83 63 94 50 102 Z", inner: "M50 96 C42.3 88.7 40.4 80 42.1 69.7 C43.7 60.6 47.1 49.3 50 37 C52.9 49.3 56.3 60.6 57.9 69.7 C59.6 80 57.7 88.7 50 96 Z", scale: 1.07, offsetY: -3, opacity: 1 },
  { outer: "M50 102 C38.5 95 33.5 84.5 35.5 69.5 C37.2 56.8 43 46.6 50 25 C57 46.6 62.8 56.8 64.5 69.5 C66.5 84.5 61.5 95 50 102 Z", inner: "M50 96 C43.3 89.8 41.5 81.6 42.8 71.6 C44.1 62.9 47.4 51.8 50 40.8 C52.6 51.8 55.9 62.9 57.2 71.6 C58.5 81.6 56.7 89.8 50 96 Z", scale: 1.04, offsetY: -2, opacity: 0.98 },
];
const IGNITION_FRAME_MS = 65;
const SUSTAIN_FRAME_MS = 180;

function setFlame(on) {
  const flame = $("flame");
  clearTimeout(BALL.flameTimer);
  flame.hidden = !on;
  if (!on) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const draw = (f) => {
    $("flame-outer-path").setAttribute("d", f.outer);
    $("flame-inner-path").setAttribute("d", f.inner);
    flame.style.transform = `translate(-50%, -72%) translateY(${f.offsetY}px) scale(${f.scale})`;
    flame.style.opacity = String(f.opacity);
  };
  let i = 0;
  const sustain = () => { draw(SUSTAIN_FRAMES[i % SUSTAIN_FRAMES.length]); i++; BALL.flameTimer = setTimeout(sustain, SUSTAIN_FRAME_MS); };
  const ignite = () => {
    if (i >= IGNITION_FRAMES.length) { i = 0; sustain(); return; }
    draw(IGNITION_FRAMES[i]); i++; BALL.flameTimer = setTimeout(ignite, IGNITION_FRAME_MS);
  };
  if (reduce) sustain(); else ignite();
}

function buildFire(tier, pos) {
  const layer = $("fire-layer");
  layer.replaceChildren();
  if (tier === 0) return;
  const i = tier / 4;
  const glow = el("div", { class: "fire-glow", "data-follow": "0" });
  glow.style.setProperty("--glow-peak", String(0.52 + i * 0.24));
  glow.style.setProperty("--glow-scale", String(1.08 + i * 0.24));
  const gd = el("div"); gd.style.width = gd.style.height = `${68 + i * 50}px`;
  glow.append(gd);
  layer.append(glow);
  if (tier >= 2) {
    const ring = el("div", { class: "fire-ring", "data-follow": "0" });
    ring.style.setProperty("--ring-scale", String(1.55 + i * 0.24));
    layer.append(ring);
  }
  if (tier >= 3) {
    const st = el("div", { class: "fire-streak", "data-follow": "0" });
    st.append(el("div"));
    layer.append(st);
  }
  if (tier >= 4) {
    for (const p of FIRE_PARTICLES) {
      const d = el("div", { class: "fire-particle", "data-follow": String(p.x), "data-y": String(p.y) });
      const size = p.size + i * 2;
      d.style.width = d.style.height = `${size}px`;
      d.style.setProperty("--dx", `${p.drift * 1.18}px`);
      d.style.setProperty("--dy", `${-p.rise * 1.18}px`);
      d.style.setProperty("--dur", `${p.duration + 0.12}s`);
      d.style.setProperty("--delay", `${p.delay}s`);
      d.style.top = `calc(50% + ${p.y}px)`;
      layer.append(d);
    }
  }
  placeFire(pos);
}

// Fire effects follow the target position directly (the source does not spring them).
function placeFire(pos) {
  for (const n of $("fire-layer").children) {
    n.style.left = `calc(${pos}% + ${Number(n.dataset.follow)}px)`;
  }
}

function renderViz(initial) {
  const T = THRESHOLD;
  const { position, isClamped } = QL.ballPosition(F.bias, T);
  const zone = QL.inTargetZone(F.bias, F.target, T);
  const tier = QL.fireTier(F.streak, FIRE_THRESHOLD);
  const fire = tier > 0;

  // labels row under the track
  const left = $("st-left"), right = $("st-right"), sig = $("st-sigma");
  left.textContent = F.bias < 0 ? QL.fmtSignedPct(F.bias, 2) : "";
  left.className = F.bias < 0 ? "neg" : "muted";
  right.textContent = F.bias > 0 ? QL.fmtSignedPct(F.bias, 2) : "";
  right.className = F.bias > 0 ? "pos" : "muted";
  sig.textContent = QL.fmtSigma(F.z);
  sig.className = `mono ${zone ? "zone" : "muted"}`;
  const big = $("big-pct");
  big.textContent = QL.fmtSignedPct(F.bias, 3);
  big.className = `big mono ${QL.biasTone(F.bias, zone)}`;
  streakMsg($("streak-msg"), F.streak);

  // ball
  const core = $("ball-core");
  core.style.background = fire ? GRADIENTS.fire : zone ? GRADIENTS.target : F.target === 1 ? GRADIENTS.green : GRADIENTS.red;
  core.classList.toggle("clamped", isClamped);
  core.style.boxShadow = fire
    ? `inset -3px -5px 10px rgba(0,0,0,0.35), inset 3px 3px 8px rgba(255,200,100,0.4), ` +
      `0 0 ${10 + tier * 5}px rgba(255,165,0,${0.28 + tier * 0.08}), 0 0 ${16 + tier * 8}px rgba(255,69,0,${0.16 + tier * 0.06}), 3px 5px 12px rgba(0,0,0,0.4)`
    : "";
  const sc = $("ball-scale");
  sc.classList.toggle("fire", fire);
  if (fire) {
    const z = zone;
    const [s0, s1] = tier >= 4 ? [z ? 1.14 : 1.04, z ? 1.17 : 1.07]
      : tier >= 3 ? [z ? 1.14 : 1.03, z ? 1.18 : 1.06]
      : tier === 2 ? [z ? 1.13 : 1.02, z ? 1.16 : 1.04]
      : [z ? 1.12 : 1.01, z ? 1.14 : 1.03];
    sc.style.setProperty("--s0", String(s0));
    sc.style.setProperty("--s1", String(s1));
    sc.style.setProperty("--pulse-dur", tier >= 4 ? "1.2s" : "1.05s");
    sc.style.setProperty("--wob", tier >= 3 ? "-0.6px" : "-0.2px");
    sc.style.transform = "";
  } else {
    sc.style.transform = `scale(${zone ? 1.15 : 1})`;
  }
  if (tier !== BALL.tier) {
    buildFire(tier, position);
    setFlame(tier >= 4);
    BALL.tier = tier;
  } else {
    placeFire(position);
  }

  // rolling: rotation follows the target (360 degrees per 100 % of movement)
  if (!BALL.init || initial) {
    BALL.spring = { x: position, v: 0 };
    BALL.target = position;
    BALL.rot = (position - 50) * 3.6; // as if the ball had rolled here from the centre
    BALL.init = true;
    placeBall();
  } else {
    BALL.rot += (position - BALL.target) * 3.6;
    BALL.target = position;
  }
  core.style.transform = `rotate(${BALL.rot}deg)`;
  core.style.filter = tier >= 2 ? `saturate(${(1 + (Math.sin((BALL.rot * Math.PI) / 180) + 1) * 0.08).toFixed(2)})` : "none";
  startBallLoop();
}

function placeBall() { $("ball").style.left = `calc(${BALL.spring.x}% - 24px)`; }

function startBallLoop() {
  if (BALL.raf) return;
  BALL.last = performance.now();
  const frame = (now) => {
    // rAF timestamps are frame-start times and can precede the performance.now() of the previous call
    const dt = Math.max(0, Math.min(0.25, (now - BALL.last) / 1000));
    BALL.last = now;
    BALL.spring = QL.springStep(BALL.spring, BALL.target, dt);
    if (QL.springSettled(BALL.spring, BALL.target)) {
      BALL.spring = { x: BALL.target, v: 0 };
      placeBall();
      BALL.raf = 0;
      return;
    }
    placeBall();
    BALL.raf = requestAnimationFrame(frame);
  };
  BALL.raf = requestAnimationFrame(frame);
}
function stopBallLoop() {
  if (BALL.raf) cancelAnimationFrame(BALL.raf);
  BALL.raf = 0; BALL.init = false; BALL.tier = -1; BALL.rot = 0;
  clearTimeout(BALL.flameTimer);
}

// ---------------------------------------------------------------- guess view
function renderGuess() {
  const perceive = F.kind === "perceive";
  const waiting = perceive && !F.trial;
  $("btn-prepare").hidden = !waiting;
  $("guess-buttons").hidden = waiting;
  $("guess-prompt").textContent = F.kind === "predict"
    ? "Guess the next bit"
    : F.trial ? "The bit is measured and hidden. Perceive it, then guess." : "Request a hidden bit, then perceive it.";
  const res = $("guess-result");
  res.replaceChildren();
  if (F.result) {
    const t = el("span", { class: F.result.hit ? "hit" : "miss" });
    t.textContent = `Bit ${F.result.bit}, you guessed ${F.result.guess}: ${F.result.hit ? "Hit" : "Miss"}`;
    res.append(t);
    if (F.result.before) {
      const tag = el("span", { class: "tag" });
      tag.textContent = "measured before guess";
      res.append(tag);
    }
  }
  const g = F.guessStats;
  if (!g) throw new Error(`no ${F.kind} stats from the server for this session`);
  $("gs-left").textContent = `${g.hits}/${g.trials} hits`;
  $("gs-center").textContent = g.hit_rate !== null ? `${(g.hit_rate * 100).toFixed(1)}% hit rate` : "no trials yet";
  $("gs-right").textContent = QL.fmtSigma(g.z_hits);
  streakMsg($("guess-streak"), F.streak);
}

// ---------------------------------------------------------------- provenance footer
function renderProvenance() {
  const s = F.session;
  const parts = [];
  if (F.hook) parts.push("TEST HOOK: injected display state, not measured data");
  else if (s) {
    const p = F.prov;
    parts.push(p ? `${p.provider}/${p.backend}` : `${s.provider}/${s.implementation}`);
    parts.push(p ? p.source : s.mode);
    if (s.num_qubits > 1) parts.push(`${s.aggregation} of ${s.num_qubits} qubits`);
    if (F.kind === "stream") parts.push(F.streaming ? "streaming" : "stream stopped");
    const j = F.job;
    if (j && !FINAL_STATES.includes(j.state)) parts.push(QL.jobStatusText(j, Date.now()));
    else if (p) parts.push(`job ${p.job_id}`);
    else if (j) parts.push(`job ${j.job_id || j.id} ${j.state}`);
    else parts.push("no job yet");
    if (F.kind === "predict" && F.result && F.result.before) parts.push("measured before guess");
  }
  $("provenance").textContent = parts.join(" · ");
  syncJobTimer();
}

// Elapsed time of the session's running job ticks once a second while that job is not final.
let jobTimer = 0;
function syncJobTimer() {
  const running = F.session && F.job && !FINAL_STATES.includes(F.job.state);
  if (running && !jobTimer) jobTimer = setInterval(() => guarded(renderProvenance), 1000);
  else if (!running) stopJobTimer();
}
function stopJobTimer() { if (jobTimer) { clearInterval(jobTimer); jobTimer = 0; } }

// ================================================================ KEYS
function wireFocus() {
  $("btn-exit").addEventListener("click", () => exitFocus());
  $("btn-stream-stop").addEventListener("click", (e) => { e.currentTarget.blur(); if (F.streaming) toggleStream(); });
  const card = $("ball-card");
  card.addEventListener("click", () => doPrimary());
  $("btn-prepare").addEventListener("click", (e) => { e.currentTarget.blur(); doPrimary(); });
  for (const b of document.querySelectorAll("[data-guess]")) {
    b.addEventListener("click", () => { b.blur(); doGuess(Number(b.dataset.guess)); });
  }
  document.addEventListener("keydown", (e) => {
    if ($("focus").hidden || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
    const k = e.key;
    if (k === " " || k === "Spacebar") { e.preventDefault(); if (!e.repeat) doPrimary(); }
    else if (k === "s" || k === "S") { if (!e.repeat && F.kind === "stream") doPrimary(); }
    else if (k === "0" || k === "1") { if (!e.repeat) doGuess(Number(k)); }
    else if (k === "f" || k === "F" || k === "Escape") { if (!e.repeat) exitFocus(); }
  });
  // Space on a focused native button fires its click on keyup: swallow it.
  document.addEventListener("keyup", (e) => { if (!$("focus").hidden && (e.key === " " || e.key === "Spacebar")) e.preventDefault(); });
}

// ================================================================ TEST HOOK (off by default)
// Only with ?hook=1 in the URL: show(target, bias, streak, n) renders the focus view with
// injected numbers so the look can be checked. It talks to no
// session, and the footer says so. Never enabled otherwise.
function installHook() {
  window.__qlHook = {
    show({ target, bias, streak, n, kind }) {
      closeSessionSse();
      Object.assign(F, {
        session: null, kind: kind || "key_press", target, bias, n, z: n ? (bias / 100) * Math.sqrt(n) : 0, streak,
        streaming: false, trial: null, result: null, guessStats: null, statsShots: 0, prov: null, job: null, busy: 0, hook: true,
      });
      $("setup").hidden = true;
      $("focus").hidden = false;
      buildFocus();
      renderFocus(true);
    },
  };
}

// ================================================================ INIT
async function init() {
  wireSetup();
  wireFocus();
  renderErrors();
  renderSetup();
  if (new URLSearchParams(location.search).get("hook") === "1") installHook();
  openGlobalSse();
  await loadProviders();
  $("prefetch-shots").value = $("provider").value === "ibm" ? 20000 : 5000;
  await loadPools();
  await loadBudget();
  renderSetup();
}
guarded(init);
