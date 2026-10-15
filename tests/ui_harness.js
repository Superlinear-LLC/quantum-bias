"use strict";
// Runs the real static/app.js against a real quantum_lite server, with a small DOM built from the
// real static/index.html. Driven by tests/test_ui_app.py; prints one JSON document on stdout.
//
//   node ui_harness.js <base_url> <scenario> '<json params>'
//
// Real: logic.js, app.js, index.html ids/data attributes, fetch (HTTP to the server), the SSE
// parsing of EventSource (over fetch). Stubbed: the DOM itself (no layout, no CSS). The harness
// reports what app.js rendered; the test decides whether that is right.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const STATIC = path.resolve(__dirname, "..", "src", "quantum_lite", "static");
const [BASE, SCENARIO, PARAMS_JSON] = process.argv.slice(2);
if (!BASE || !SCENARIO) throw new Error("usage: ui_harness.js <base_url> <scenario> <json params>");
const PARAMS = JSON.parse(PARAMS_JSON || "{}");
const realFetch = globalThis.fetch;

function die(err) {
  process.stderr.write(`harness failure: ${err && err.stack ? err.stack : err}\n`);
  process.exit(2);
}
process.on("unhandledRejection", die);
process.on("uncaughtException", die);

// ---------------------------------------------------------------- minimal DOM
const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

class Text {
  constructor(data) { this.data = data; this.children = []; }
  get textContent() { return this.data; }
}

class El {
  constructor(tag, attrs) {
    attrs = attrs || {};
    this.tagName = tag.toUpperCase();
    this.attrs = {};
    this.children = [];
    this.parent = null;
    this._text = "";
    this.listeners = {};
    this.dataset = {};
    const self = this;
    this.style = { setProperty(k, v) { self.style[k] = String(v); } };
    this.className = "";
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    for (const [k, v] of Object.entries(attrs)) this.setAttribute(k, v);
    if ("hidden" in attrs) this.hidden = true;
    if ("value" in attrs) this.value = attrs.value;
  }
  setAttribute(k, v) {
    v = String(v);
    this.attrs[k] = v;
    if (k === "class") this.className = v;
    if (k.startsWith("data-")) this.dataset[camel(k.slice(5))] = v;
  }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  get classList() {
    const self = this;
    const toks = () => self.className.split(/\s+/).filter(Boolean);
    return {
      contains: (c) => toks().includes(c),
      add(c) { if (!toks().includes(c)) self.className = [...toks(), c].join(" "); },
      remove(c) { self.className = toks().filter((t) => t !== c).join(" "); },
      toggle(c, force) {
        const on = force === undefined ? !toks().includes(c) : force;
        if (on) this.add(c); else this.remove(c);
        return on;
      },
    };
  }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get firstChild() { return this.children[0] || null; }
  append(...kids) {
    for (const k of kids) {
      const node = typeof k === "string" ? new Text(k) : k;
      node.parent = this;
      this.children.push(node);
    }
  }
  replaceChildren(...kids) { this._text = ""; this.children = []; this.append(...kids); }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatch(type, ev) { for (const fn of this.listeners[type] || []) fn(ev); }
  click() { this.dispatch("click", { currentTarget: this, target: this, preventDefault() {} }); }
  blur() {}
  focus() {}
  matches(sel) {
    if (sel.startsWith("#")) return this.attrs.id === sel.slice(1);
    if (sel.startsWith(".")) return this.classList.contains(sel.slice(1));
    const m = /^\[([\w-]+)\]$/.exec(sel);
    if (m) return m[1] in this.attrs;
    return this.tagName === sel.toUpperCase();
  }
  closest(sel) {
    for (let n = this; n; n = n.parent) if (n instanceof El && n.matches(sel)) return n;
    return null;
  }
  *walk() {
    for (const c of this.children) { if (c instanceof El) { yield c; yield* c.walk(); } }
  }
  querySelectorAll(sel) { return [...this.walk()].filter((e) => e.matches(sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

const VOID = new Set(["meta", "link", "input", "img", "br", "hr"]);
function parseHtml(html) {
  const root = new El("root");
  const stack = [root];
  const byId = new Map();
  const tag = /<!--[\s\S]*?-->|<!doctype[^>]*>|<(\/?)([a-zA-Z][\w:-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/gi;
  let m;
  while ((m = tag.exec(html))) {
    if (!m[2]) continue;
    const name = m[2].toLowerCase();
    if (m[1]) {
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].tagName === name.toUpperCase()) { stack.length = i; break; }
      }
      continue;
    }
    const attrs = {};
    const re = /([\w:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;
    let a;
    const raw = m[3].replace(/\/\s*$/, "");
    while ((a = re.exec(raw))) attrs[a[1]] = a[2] ?? a[3] ?? a[4] ?? "";
    const e = new El(name, attrs);
    stack[stack.length - 1].append(e);
    if (attrs.id) {
      if (byId.has(attrs.id)) throw new Error(`index.html: duplicate id ${attrs.id}`);
      byId.set(attrs.id, e);
    }
    if (!VOID.has(name) && !/\/\s*$/.test(m[3])) stack.push(e);
  }
  return { root, byId };
}

const { root, byId } = parseHtml(fs.readFileSync(path.join(STATIC, "index.html"), "utf8"));
const docListeners = {};
globalThis.document = {
  getElementById(id) {
    const e = byId.get(id);
    if (!e) throw new Error(`app.js asked for #${id}, which index.html does not have`);
    return e;
  },
  createElement: (tag) => new El(tag),
  createElementNS: (_ns, tag) => new El(tag),
  querySelectorAll: (sel) => root.querySelectorAll(sel),
  querySelector: (sel) => root.querySelector(sel),
  addEventListener(type, fn) { (docListeners[type] = docListeners[type] || []).push(fn); },
  activeElement: null,
};
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.location = { search: "" };
globalThis.matchMedia = () => ({ matches: false });
globalThis.requestAnimationFrame = (fn) => setTimeout(() => fn(performance.now()), 16);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
globalThis.fetch = (url, opts) => realFetch(BASE + url, opts);

// EventSource over fetch: the same wire format parsing a browser does.
class EventSource {
  constructor(url) {
    this.url = url; this.readyState = 0; this.onopen = null; this.onerror = null;
    this._l = {}; this._ac = new AbortController();
    this._run().catch((err) => { if (this.readyState !== 2) die(err); });
  }
  addEventListener(type, fn) { (this._l[type] = this._l[type] || []).push(fn); }
  close() { this.readyState = 2; this._ac.abort(); }
  async _run() {
    const r = await realFetch(BASE + this.url, { signal: this._ac.signal, headers: { Accept: "text/event-stream" } });
    if (!r.ok) throw new Error(`SSE ${this.url}: HTTP ${r.status}`);
    this.readyState = 1;
    if (this.onopen) this.onopen({});
    const dec = new TextDecoder();
    let buf = "";
    for await (const chunk of r.body) {
      buf += dec.decode(chunk, { stream: true }).replace(/\r\n/g, "\n");
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) { this._emit(buf.slice(0, i)); buf = buf.slice(i + 2); }
    }
    if (this.readyState !== 2) throw new Error(`SSE ${this.url} ended`);
  }
  _emit(block) {
    let type = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue;
      if (line.startsWith("event:")) type = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) return;
    for (const fn of this._l[type] || []) fn({ data: data.join("\n") });
  }
}
EventSource.CONNECTING = 0; EventSource.OPEN = 1; EventSource.CLOSED = 2;
globalThis.EventSource = EventSource;

// ---------------------------------------------------------------- load the real scripts
function load(name) { vm.runInThisContext(fs.readFileSync(path.join(STATIC, name), "utf8"), { filename: name }); }
load("logic.js");
load("app.js");
const peek = (expr) => vm.runInThisContext(expr); // read app.js state (script-scope consts)

// ---------------------------------------------------------------- driving helpers
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, what, ms = 15000) {
  const t0 = Date.now();
  for (;;) {
    const v = fn();
    if (v) return v;
    if (Date.now() - t0 > ms) throw new Error(`timed out waiting for ${typeof what === "function" ? what() : what}`);
    await sleep(5);
  }
}
const $ = (id) => document.getElementById(id);
const text = (id) => $(id).textContent;
function key(k) {
  const ev = { key: k, target: { tagName: "BODY" }, metaKey: false, ctrlKey: false, altKey: false, repeat: false, preventDefault() {} };
  for (const fn of docListeners.keydown || []) fn(ev);
}
function snap(label) {
  return {
    label,
    title: text("focus-title"),
    big: text("big-pct"), sigma: text("st-sigma"), left: text("st-left"), right: text("st-right"),
    streak: text("streak-msg"),
    ballTarget: peek("BALL.target"), trackLeft: text("lbl-left"), trackRight: text("lbl-right"),
    prompt: text("guess-prompt"), result: text("guess-result"),
    gsLeft: text("gs-left"), gsCenter: text("gs-center"), gsRight: text("gs-right"), guessStreak: text("guess-streak"),
    guessButtonsHidden: $("guess-buttons").hidden, prepareHidden: $("btn-prepare").hidden,
    provenance: text("provenance"),
    alerts: $("alerts-focus").children.map((c) => c.querySelector("pre").textContent),
    shots: peek("F.statsShots"),
    barHidden: $("stream-bar").hidden, barText: text("stream-status"), barClass: $("stream-status").className,
    stopHidden: $("btn-stream-stop").hidden,
  };
}
const clickData = (attr, value) => {
  const b = document.querySelectorAll(`[data-${attr}]`).find((e) => e.dataset[attr] === value);
  if (!b) throw new Error(`no [data-${attr}=${value}] in index.html`);
  b.click();
};

function setField(id, value) { $(id).value = String(value); $(id).dispatch("input", {}); }
function chooseProvider(id) { $("provider").value = id; $("provider").dispatch("change", {}); }

async function pickSetup(setup) {
  await until(() => peek("S.budget !== null"), "app.js init (providers, pools, budget)");
  if (setup.goal) clickData("goal", setup.goal);
  if (setup.sub) clickData("sub", setup.sub);
  if (setup.target !== undefined) clickData("target", String(setup.target));
  clickData("source", PARAMS.source || "live"); // acquisition is scripted by the server side of the test
  if (PARAMS.provider) chooseProvider(PARAMS.provider);
  if (PARAMS.seconds !== undefined) setField("stream-seconds", PARAMS.seconds);
  if (PARAMS.max_jobs !== undefined) setField("stream-max-jobs", PARAMS.max_jobs);
}

async function startSession(setup) {
  await pickSetup(setup);
  $("btn-start").click();
  await until(() => !$("focus").hidden && peek("F.session !== null"),
    () => `the focus view; providers ${peek("JSON.stringify(S.providers)")} select=${$("provider").value}; setup alerts:${JSON.stringify($("alerts-setup").children.map((c) => c.textContent))}`);
  return peek("F.session.id");
}
const settled = () => peek("F.busy") === 0;

const scenarios = {
  async press() {
    const sid = await startSession({ goal: "affect", sub: "key_press", target: PARAMS.target });
    const snaps = [];
    for (let k = 1; k <= PARAMS.presses; k++) {
      key(" ");
      await until(() => settled() && peek("F.statsShots") >= k, `press ${k} applied`);
      await sleep(10);
      snaps.push(snap(`press ${k}`));
    }
    return { sid, snaps };
  },
  // C2: a slow request of session A must not leave #focus dimmed in session B.
  async busy_after_exit() {
    const sidA = await startSession({ goal: "affect", sub: "key_press", target: 1 });
    const cls = () => $("focus").className;
    const out = { sidA, cases: {} };
    key(" "); // the server holds this press for PARAMS.delay_ms
    if (PARAMS.exit_before_timer) {
      key("f");
      await until(() => $("focus").hidden, "focus exited");
    } else {
      await until(() => $("focus").classList.contains("busy"), "the busy class after 250 ms");
      out.cases.busyWhileSlow = true;
      key("f");
      await until(() => $("focus").hidden, "focus exited");
    }
    out.cases.afterExit = cls();
    out.sidB = await startSession({ goal: "affect", sub: "key_press", target: 1 });
    out.cases.afterOpenB = cls();
    await sleep(400); // past A's 250 ms timer, A's request still pending
    out.cases.timerWindow = cls();
    await sleep(PARAMS.delay_ms + 300); // A's response arrives
    out.cases.afterAResolved = cls();
    return out;
  },
  async press_error() {
    const sid = await startSession({ goal: "affect", sub: "key_press", target: PARAMS.target });
    key(" ");
    await until(() => settled() && $("alerts-focus").children.length > 0, "the error alert");
    return { sid, snaps: [snap("refused press")] };
  },
  async stream_error() {
    const sid = await startSession({ goal: "affect", sub: "stream", target: PARAMS.target });
    key("s");
    await until(() => peek("F.streaming"), "stream started");
    // the feed shows the failure; the session read that follows sets streaming false
    await until(() => !peek("F.streaming") && settled(), "the failed stream noticed");
    await sleep(100);
    return { sid, snaps: [snap("stream failed")] };
  },
  async stream() {
    const sid = await startSession({ goal: "affect", sub: "stream", target: PARAMS.target });
    const snaps = [];
    key("s");
    await until(() => peek("F.streaming"), "stream started");
    let seen = 0;
    await until(() => {
      const n = peek("F.statsShots");
      if (n !== seen) { seen = n; snaps.push(snap(`shots ${n}`)); }
      return n >= PARAMS.total;
    }, `${PARAMS.total} streamed shots`);
    await sleep(30);
    snaps[snaps.length - 1] = snap(`shots ${seen}`); // events of the last batch fully applied
    key("s");
    await until(() => !peek("F.streaming") && settled(), "stream stopped");
    snaps.push(snap("stopped"));
    return { sid, snaps };
  },
  // The Stream limits fields: shown for Stream only, Max IBM jobs for ibm only, the estimate, the
  // validation of bad values (nothing is created), and what a valid Start sends.
  async stream_setup() {
    await until(() => peek("S.budget !== null"), "app.js init");
    const out = { hidden: {} };
    clickData("goal", "affect");
    clickData("source", "live");
    out.hidden.keyPress = $("group-stream").hidden;
    clickData("sub", "stream");
    out.hidden.stream = $("group-stream").hidden;
    out.hidden.jobsRowSim = $("stream-jobs-row").hidden;
    out.defaults = { seconds: $("stream-seconds").value, jobs: $("stream-max-jobs").value };
    chooseProvider("ibm");
    out.hidden.jobsRowIbm = $("stream-jobs-row").hidden;
    out.estimateDefault = text("stream-estimate");
    setField("stream-max-jobs", 7);
    out.estimateSeven = text("stream-estimate");
    setField("stream-shots", 2000);
    out.estimateBigShots = text("stream-estimate");
    setField("stream-shots", 20);
    clickData("source", "buffered");
    out.estimateBuffered = text("stream-estimate");
    clickData("source", "live");
    clickData("goal", "guess");
    out.hidden.guess = $("group-stream").hidden;
    clickData("goal", "affect");
    for (const [secs, jobs] of [["0", "7"], ["abc", "7"], ["2.5", "7"], ["30", "0"], ["30", ""]]) {
      setField("stream-seconds", secs); setField("stream-max-jobs", jobs);
      $("btn-start").click();
      await until(() => !peek("S.starting"), "start refused");
      await sleep(20);
      out.alerts = out.alerts || [];
      out.alerts.push($("alerts-setup").children.map((c) => c.querySelector("pre").textContent).pop());
      out.focusHiddenAfterBad = $("focus").hidden;
    }
    setField("stream-seconds", 45); setField("stream-max-jobs", 7);
    $("btn-start").click();
    await until(() => !$("focus").hidden && peek("F.session !== null"), "the focus view");
    out.sid = peek("F.session.id");
    return out;
  },
  // A stream ended by its limit: countdown while running, the limit message after, no red alert.
  async stream_limit() {
    const sid = await startSession({ goal: "affect", sub: "stream", target: 1 });
    const out = { sid, running: [], stopped: null };
    key("s");
    await until(() => peek("F.streaming"), "stream started");
    const seen = new Set();
    await until(() => {
      if (peek("F.streaming")) {
        const s = snap("running");
        seen.add(s.barText);
        if (!out.running.length) out.running.push(s);
      }
      return !peek("F.streaming");
    }, "the stream to be stopped by its limit", 20000);
    await sleep(100);
    out.seen = [...seen];
    out.stopped = snap("limit stop");
    out.timerCleared = peek("streamTimer") === 0;
    return out;
  },
  // The clickable Stop control stops the stream; no limit message is shown for it.
  async stream_stop_button() {
    const sid = await startSession({ goal: "affect", sub: "stream", target: 1 });
    key("s");
    await until(() => peek("F.streaming"), "stream started");
    await until(() => peek("F.statsShots") >= 20, "a first batch");
    const running = snap("running");
    $("btn-stream-stop").click();
    await until(() => !peek("F.streaming") && settled(), "stream stopped by the Stop control");
    await sleep(100);
    return { sid, running, stopped: snap("stopped by button") };
  },
  // No server event arrives after the first batch: the countdown must tick by itself.
  async stream_countdown_idle() {
    const sid = await startSession({ goal: "affect", sub: "stream", target: 1 });
    key("s");
    await until(() => peek("F.statsShots") >= 20, "a first batch");
    await sleep(100);
    const first = snap("first");
    await sleep(2200);
    const later = snap("later");
    key("s");
    await until(() => !peek("F.streaming") && settled(), "stream stopped");
    return { sid, first, later };
  },
  async predict() {
    const sid = await startSession({ goal: "guess", sub: "predict" });
    const snaps = [];
    for (let k = 1; k <= PARAMS.guesses.length; k++) {
      key(String(PARAMS.guesses[k - 1]));
      await until(() => settled() && peek("F.guessStats && F.guessStats.trials") >= k, `guess ${k} applied`);
      await sleep(10);
      snaps.push(snap(`guess ${k}`));
    }
    return { sid, snaps };
  },
  async perceive() {
    const sid = await startSession({ goal: "guess", sub: "perceive" });
    const snaps = [snap("before prepare")];
    key(" ");
    await until(() => settled() && peek("F.trial !== null"), "hidden bit prepared");
    snaps.push(snap("prepared"));
    key(String(PARAMS.guess));
    await until(() => settled() && peek("F.trial === null") && peek("F.guessStats && F.guessStats.trials") >= 1, "reveal applied");
    await sleep(10);
    snaps.push(snap("revealed"));
    return { sid, snaps };
  },
};

(async () => {
  const run = scenarios[SCENARIO];
  if (!run) throw new Error(`unknown scenario ${SCENARIO}`);
  const out = await run();
  process.stdout.write(JSON.stringify(out));
  process.exit(0);
})().catch(die);
