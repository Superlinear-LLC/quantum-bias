# Quantum Bias Lab: design

## Goal

A light, laptop-runnable bias lab. Anyone who clones the repo runs it on a **local simulator**
with no account. With an IBM Quantum API key the same app runs against a **real IBM QPU**.

## Hard requirements

1. Runs on a laptop: macOS (arm64/x86_64), Linux, Windows. One process. No Docker, Node,
   Postgres, Redis, Consul, AWS or Google sign-in. No bash-only scripts on the run path.
2. Two providers:
   - `sim`: local qiskit-aer. Default. Always available.
   - `ibm`: real QPU through qiskit-ibm-runtime. Available only when a token is configured.
3. Two acquisition modes, **both** required, selectable per session. The API can switch them
   while a session runs (`PATCH /api/sessions/{id}`); the UI picks the mode at setup only
   (the UI is two screens: setup and focus view):
   - **buffered**: bits come from a pre-fetched pool of real measurements (one or more large
     jobs run earlier). Smooth for a stage demo.
   - **live**: every interaction (and every stream cycle) submits a fresh job and waits for
     it. The UI shows real job state (queued, queue position if known, running, elapsed).
4. Provenance is always visible: each bit/batch carries `job_id`, `source`
   (`live`|`buffered`), `provider`, `backend`, `measured_at`. Stats are kept per mode segment
   as well as for the whole session, so buffered bits never pass as live ones.
5. Fail fast. No silent fallbacks: never fall back from `ibm` to `sim`, from live to buffered,
   or from an empty pool to anything. Errors propagate to the API response and the UI with the
   real message. No fabricated or placeholder bits, ever.

## Stack

- Python `>=3.10,<3.14`. Package `quantum_lite` in `src/quantum_lite/`, built with hatchling, locked in `uv.lock`.
- Pinned: `qiskit==1.4.5`, `qiskit-aer==0.17.2`, `qiskit-ibm-runtime==0.44.1` (last runtime
  line on qiskit 1.x; SamplerV2 import is `from qiskit_ibm_runtime import SamplerV2`).
  Plus `fastapi`, `uvicorn`, `numpy`, `python-dotenv`. Dev group: `pytest`, `httpx`.
- Web UI: static `index.html` + `app.js` + `style.css` under
  `src/quantum_lite/static/`, served by FastAPI at `/`. Vanilla JS, no build step,
  no CDN (works offline).
- Server binds `127.0.0.1` by default, port `8765`. `--host` flag exists; binding anything
  else prints a warning (single-user app, no auth).

## Entry points

- `uv run quantum-lite` (console script) == `python -m quantum_lite`. Flags: `--host`,
  `--port`, `--open` (open browser), `--env-file`.
- `uv run quantum-lite prefetch --provider ibm --qubits 5 --shots 20000` fills a pool on disk
  before going on stage, prints job id, shot count, estimated QPU seconds, then exits.
- `uv run quantum-lite pools` lists pools on disk.

## Configuration (env vars, optional `.env` in the project root)

| Var | Default | Meaning |
|---|---|---|
| `IBM_QUANTUM_TOKEN` (alias `QISKIT_IBM_TOKEN`) | unset | API key. Unset → `ibm` provider listed as `configured: false`. |
| `IBM_QUANTUM_INSTANCE` | unset | Optional CRN / instance name. |
| `IBM_QUANTUM_CHANNEL` | `ibm_quantum_platform` | Runtime channel. |
| `IBM_BACKEND` | unset | Fixed device name; unset → `least_busy(operational=True, simulator=False)` resolved once per process. |
| `IBM_MAX_SHOTS_PER_JOB` | `20000` | Ceiling per job. Requests above it → 400. |
| `IBM_MAX_TOTAL_SHOTS` | `200000` | Per-process ceiling across all IBM jobs. Exceeding → 429 with the numbers. |
| `IBM_MAX_JOBS` | `100` | Per-process job count ceiling. |
| `SIM_NOISE` | `none` | `none` = ideal AerSimulator; `fake` = noise model from a qiskit-ibm-runtime fake backend (pick one that exists in 0.44.1, ≤ 27 qubits, e.g. `FakeTorontoV2`/`FakeMelbourneV2`; verify by import). |
| `SIM_LATENCY_SECONDS` | `0` | Artificial delay per sim job, so live mode can be shown without a key. Reported as such in job events (`simulated_latency: true`). |
| `QL_DATA_DIR` | `.data/` in the project root (gitignored) | Pools on disk. |

Token handling: passed straight to `QiskitRuntimeService(channel=..., token=..., instance=...)`.
**Never** call `save_account`. Never log, print, return or embed the token (also not in
exception messages, `repr`, `/api/providers`, or the static UI). `.env` and `.data/` are
gitignored; `.env.example` documents the vars with empty values.

## Domain

Vocabulary: the glossary is `CONTEXT.md`.

Circuit: `independent_qubits` — H on each of `n` qubits, measure into a classical register
named `c` of size `n`. `1 <= n <= 16` (UI default 5).

Bit order: a Measurement is an `(shots, n)` uint8 matrix, column `j` = logical qubit `j`.
Qiskit bitstrings are little-endian (qubit 0 rightmost) and must be reversed. A test with an
`X` on qubit 0 only proves it (column 0 all ones, others ≈ 50%).

IBM: transpile with `generate_preset_pass_manager(optimization_level=1, backend=backend,
initial_layout=physical_qubits)`; `physical_qubits` = the `n` qubits with lowest measure error
from `backend.target["measure"]` (ties → lowest index), chosen once per (backend, n), reported
in session info. `SamplerV2(mode=backend)` — job mode (the IBM Open Plan has no sessions). Job state
polled with `job.status()` every ~2 s off the event loop; `queue_position` not available in
all versions → report `null`, never invent one. No calibration jobs, no hidden jobs: every
QPU job is one the user caused, and each is listed in `/api/jobs`.

Sim: same circuit, `AerSimulator` (ideal) or `AerSimulator.from_backend(fake)` (noisy) with the
same physical-qubit selection rule. Runs in a thread pool, never blocking the event loop.

### Aggregation (one bit from one shot)

`aggregation` ∈ `none` (qubit 0), `xor` (XOR of the n bits), `majority` (requires odd `n`;
even `n` → 400 at session create). Used for single-bit interactions. Stream stats count
every raw bit (shots × n) and also the aggregated bit per shot.

### Stats (per session, per segment, per mode)

For `N` bits with `k` ones:
- `bias_pct = (k - N/2) / (N/2) * 100`
- `z = (k - N/2) / (sqrt(N)/2)` (0 when N = 0)
- sliding window (last `W` = 1000 bits) `bias_pct` and `z`
- per-qubit ones fraction (shows readout bias on hardware)
- Predict/Perceive: trials, hits, hit rate, `z_hits = (hits - T/2)/(sqrt(T)/2)`, current and
  best streak.

### Pools (buffered mode)

- A pool is keyed by `(provider, implementation, n_qubits)`; contents: a list of measurement
  blocks `{job_id, backend, physical_qubits, measured_at, shots, bits}` stored as JSON in
  `QL_DATA_DIR/pools/<key>.json` (bits packed as base64 of `np.packbits`). Atomic write
  (tmp + rename). Loaded at startup.
- Consumption is in order, one shot at a time; a consumed shot is never reused (cursor
  persisted).
- `POST /api/pools/prefetch` submits one job of `shots` and appends on completion; it is
  async (returns a job id; progress on `/api/events`). A prefetch that is still resolving its
  device when server shutdown begins is refused with 503 `shutting_down` and submits no job.
- Empty pool in buffered mode → 409 `pool_empty` with the pool key; no auto-fallback.
  Optional `auto_refill` session flag (sim only; `ibm` refuses it with 400 `auto_refill_ibm`, since every
  refill there is a billed QPU job, so ibm pools are refilled only by `POST /api/pools/prefetch`)
  submits a prefetch when the pool drops below `refill_below` shots; while it runs, requests still 409
  when the pool is empty.
  A failed auto_refill job switches auto_refill off for that session and is never retried automatically.

### Interactions

- **stream**: start/stop. Live: one job of `shots_per_job` at a time (pipeline depth 1),
  next job starts when the previous result arrives. Depth 1 is enforced at the permit: a live
  cycle that finds the session's previous stream job still in flight (a stopped run's job, which
  cannot be recalled) waits for it, then reads the session's mode afresh, so a switch to buffered
  during the wait submits no live job and a session never has two stream jobs in flight; a
  stream start or a switch to live is not refused for it. Buffered: emits `shots_per_job` shots
  from the pool every `interval_ms` (default 500). Each cycle emits a `batch` event with
  provenance + updated stats. A stop never drops bits already paid for: a live job in flight
  finishes and its bits go to the pool; shots a buffered cycle is already taking from the pool
  are taken, recorded and published as its `batch` before the stop completes. Server shutdown
  is the exception: sessions are not kept across it, and a buffered key press, predict or
  perceive whose pool take is still running when the 5 s shutdown grace ends is cancelled; its
  shot stays consumed and is recorded nowhere. Shutdown does not wait for jobs either: a live,
  prefetch or auto_refill job still running when shutdown begins is not pooled, even if it
  finishes on the device (on `ibm` it is still billed); it stays listed in `/api/jobs`. **Bounded by the server**: the loop stops itself
  when `stream_seconds` of wall time have passed (checked before each cycle; a live job in flight
  at the deadline is not waited for, it finishes and its bits go to the pool, as for
  `/stream/stop`; a cycle still waiting for the previous job at the deadline submits nothing) or
  when the limit refuses a job: in live mode when `stream_max_jobs` jobs of this run were
  submitted, and in buffered mode (sim only) when it refuses an auto_refill the stream caused.
  A limit stop is not an error: `stream.stopped_reason` is `limit_seconds: ...` or
  `limit_jobs: ...` and the `stream_stopped` event is published.
- **key press**: one aggregated bit.
- **predict**: client sends guess (0/1) first, then the bit is acquired. Live: the job is
  submitted after the guess arrives (bit generated after the guess). Buffered: next pool shot;
  the response and UI mark it `measured_before_guess: true`.
- **perceive**: `prepare` acquires and hides a bit (server keeps it; response has only a
  `trial_id`), then `reveal` with a guess returns the bit and the hit. One pending prepare per
  session; a second prepare → 409.

All interactions work on both providers and both modes. A live single-bit interaction runs
one job of `shots = 1`.

## HTTP API (JSON; errors `{"error": code, "detail": msg}`)

- `GET /api/health`
- `GET /api/providers` → `[{id, variant, implementation, configured, available, reason}]`.
  `ibm` without token → `configured: false, reason: "IBM_QUANTUM_TOKEN not set"`. Never
  contacts IBM at startup; `implementation` for `ibm` resolves on first use.
- `POST /api/sessions` `{provider, mode, num_qubits, aggregation, shots_per_job, interval_ms,
  auto_refill, refill_below, stream_seconds, stream_max_jobs}` → session. `stream_seconds`
  (required, integer 1..3600): longest a stream run may last. `stream_max_jobs` (integer
  1..1000): IBM jobs one stream run may cause (its live cycle jobs, plus on sim the auto_refill jobs it triggers; counted in `stream.jobs_used`, any provider's jobs, only `ibm` bills); required for every `ibm` session (the mode can be
  patched to live later). Missing or invalid → 400 with the reason. The session's `stream` object:
  `{running, error, stopped_reason, seconds_limit, seconds_left, jobs_limit, jobs_used}`.
- `GET /api/sessions/{id}`; `PATCH /api/sessions/{id}` `{mode}` (switch live ↔ buffered; starts
  a new segment)
- `POST /api/sessions/{id}/stream/start`, `/stream/stop`
- `POST /api/sessions/{id}/press`
- `POST /api/sessions/{id}/predict` `{guess}`
- `POST /api/sessions/{id}/perceive/prepare`, `/perceive/reveal` `{trial_id, guess}`
- `GET /api/sessions/{id}/events` — SSE: `batch`, `bit`, `job`, `stats`, `stream` (state of the run: started, a live job submitted),
  `stream_stopped` (a limit ended the run), `error`, heartbeat
  every 15 s. Bounded per-client queue; drops counted and reported in the next event.
- `GET /api/pools`, `POST /api/pools/prefetch` `{provider, num_qubits, shots}`,
  `DELETE /api/pools/{key}`
- `GET /api/jobs` — every job this process submitted: id, provider, backend, shots, state,
  submitted/finished times, estimated QPU seconds (`2 + 0.00035 * shots`, labelled estimate).
- `GET /api/budget` — IBM shots/jobs used vs ceilings.
- `GET /api/events` — global SSE for job and pool events.

## UI

Two screens, dark theme (Geist font, bundled). No qubit calibration; the full-screen focus view
is the main surface.

- Setup page: Interaction Goal (Affect: Key Press / Stream; Guess: Predict / Perceive), Target
  0/1, Quantum Source (disabled `ibm` with the reason when not configured), Buffered / Live,
  Parallel Aggregation (Disabled / XOR / Majority) with qubits, a pool line with Prefetch and,
  for `ibm`, estimated QPU seconds; with Stream chosen, Stream duration (seconds, default 30)
  and, for `ibm`, Max IBM jobs (default 5) with the estimated QPU seconds for that many jobs;
  Start Training.
- Focus view: ball on the gradient track, big bias %, sigma, streak messages and fire tiers,
  SPACE / F hints, exit. Guess modes show 0/1 buttons and hit stats, with `measured before
  guess` where it applies. Footer: provenance of the latest bits and the session's job (id,
  state, queue position or "not reported", elapsed). Stream: a countdown (seconds left, and
  `jobs N/M` on live `ibm`), a Stop stream button next to S / SPACE, and after a limit stop
  "Stream stopped: N s limit reached" (or "N job limit reached"), not shown as an error.
- All errors shown verbatim. No external assets; works offline.

## Tests (`tests/`, pytest; `uv run pytest` passes with zero failures, warnings as errors)

- stats math against hand-computed values (incl. N = 0)
- aggregation: xor, majority, majority with even n rejected
- bit order: X on qubit 0 (sim) → column 0 all ones; column 1 ≈ 0.5 (tolerance from shots)
- sim provider: shapes, physical qubits, `SIM_LATENCY_SECONDS` reported
- pools: write/load round trip, cursor persistence, no reuse, empty → 409, atomic write
- API (FastAPI TestClient, sim): every interaction × both modes; mode switch creates a segment;
  perceive double-prepare → 409; predict buffered marks `measured_before_guess`
- IBM provider with a fake `QiskitRuntimeService`/`SamplerV2` boundary: token passed, never
  `save_account`; token absent from logs (caplog) and from `/api/providers`; least_busy used when
  `IBM_BACKEND` unset; ceilings → 400/429; job states mapped; no IBM call at startup
- no-fallback tests: IBM error surfaces (not sim bits); empty pool does not go live
- live IBM smoke test, skipped unless `QL_LIVE_IBM=1` and a token are set (never run in CI)

## Out of scope

Persistence of sessions across restarts, accounts/auth, calibration and qubit ranking,
best-only selection, analytics, Ramsey/spin echo, other vendors.

## Cost controls

The IBM Open Plan allows 10 QPU-minutes (600 s) per rolling 28 days. The app shows the estimate
`2 s + 0.00035 s x shots` per job, which is IBM's published estimate, not a measurement, so real
usage can differ. A 20000-shot prefetch is about 9 s; a 1-shot live job is about 2 s.

- Process ceilings: `IBM_MAX_SHOTS_PER_JOB`, `IBM_MAX_TOTAL_SHOTS`, `IBM_MAX_JOBS`. They count
  this process only (they reset on restart) and know nothing of what the account already spent.
- Stream limits contract: every stream is bounded by the server. `stream_seconds` (required, 1 to
  3600) limits wall time; on `ibm`, `stream_max_jobs` (required, 1 to 1000) limits the IBM jobs one
  stream run causes: its live cycle jobs (and, on sim, the auto-refill jobs the stream triggers;
  `ibm` has no auto-refill). It is not the process-wide count, and not refills that Key Press,
  Predict or Perceive trigger. Nothing the stream causes is submitted at or after the deadline or
  beyond the limit; a job the limit refuses is not submitted and the stream stops with
  `limit_jobs`. The check sits at the submit itself, so a stalled event loop cannot push a job
  past the deadline.
- The UI estimate next to Max IBM jobs is the worst case: jobs x (2 s + 0.00035 s x shots).
- A stream in live mode submits one job per cycle, so each cycle costs at least the 2 s base:
  avoid live streaming on a real QPU.
- Stopping a live stream, by hand or at a limit, cannot cancel a job already on the QPU; it
  still runs and still counts, and its bits are pooled.
