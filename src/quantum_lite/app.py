"""FastAPI app. Errors are `{"error": code, "detail": message}` with the real message."""
from __future__ import annotations

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, StrictInt

from quantum_lite import __version__
from quantum_lite.budget import IbmBudget
from quantum_lite.config import Settings
from quantum_lite.errors import AppError
from quantum_lite.events import EventBus, sse_stream
from quantum_lite.jobs import ESTIMATE_BASE_SECONDS, ESTIMATE_FORMULA, ESTIMATE_SECONDS_PER_SHOT, JobRunner
from quantum_lite.pool_feed import PoolFeeder
from quantum_lite.pools import PoolStore
from quantum_lite.providers.base import Provider
from quantum_lite.providers.ibm import IbmProvider
from quantum_lite.providers.sim import SimProvider
from quantum_lite.sessions import SessionManager

log = logging.getLogger("quantum_lite.app")
STATIC_DIR = Path(__file__).parent / "static"
# Host headers accepted when bound to a local address: a DNS-rebinding page reaches the API
# under its own hostname, and is refused. Binding elsewhere is a warned choice.
ALLOWED_LOCAL_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost")
# "::1" is deliberately not local here: Starlette's host check splits on ":" and cannot match
# "[::1]:port", so it would refuse every request. It gets the non-local warning instead.
LOCAL_BIND_HOSTS = ("127.0.0.1", "localhost")


class _RevalidatedStaticFiles(StaticFiles):
    """Static UI files that the browser revalidates on every load (ETag makes that cheap).

    Without this, Chrome reuses a heuristically cached app.js/style.css after an upgrade.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionCreate(_Body):
    provider: str = "sim"
    mode: str = "buffered"
    num_qubits: int = 5
    aggregation: str = "none"
    shots_per_job: int = 100
    interval_ms: int = 500
    auto_refill: bool = False
    refill_below: int = 0
    refill_shots: int | None = None
    # Required by the manager (400 with the reason when missing), strict so true or 2.5 is refused.
    stream_seconds: StrictInt | None = None
    stream_max_jobs: StrictInt | None = None


class SessionPatch(_Body):
    mode: str


class Guess(_Body):
    guess: Literal[0, 1]


class Reveal(_Body):
    trial_id: str
    guess: Literal[0, 1]


class Prefetch(_Body):
    provider: str = "sim"
    num_qubits: int = 5
    shots: int = 20000


def build_providers(settings: Settings) -> dict[str, Provider]:
    return {"sim": SimProvider(settings), "ibm": IbmProvider(settings)}


_POOL_KEY = re.compile(r"^[A-Za-z0-9_-]+__[A-Za-z0-9_-]+__\d+$")


def create_app(settings: Settings, providers: dict[str, Provider] | None = None,
               bind_host: str = "127.0.0.1") -> FastAPI:
    bus = EventBus()
    budget = IbmBudget(settings)
    pools = PoolStore(settings.data_dir)
    providers = providers if providers is not None else build_providers(settings)
    runner = JobRunner(bus, budget, settings.sim_latency_seconds)
    feeder = PoolFeeder(providers, pools, runner, bus, budget)  # the same PoolStore the sessions take from
    mgr = SessionManager(providers, runner, pools, bus, budget, feeder)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await begin_shutdown()

    async def begin_shutdown() -> None:
        """Stop streams, end every SSE response, tell IBM polling threads to leave. Idempotent.

        The server calls this before it waits for connections to close (cli.make_server): an
        SSE response never ends by itself, so waiting first would keep the streams running.
        """
        bus.close_all()
        await mgr.shutdown()

    # The default /docs and /redoc pages load Swagger UI / ReDoc from a CDN: blank offline.
    # The schema itself is at /openapi.json.
    app = FastAPI(title="Quantum Bias Lab", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    if bind_host in LOCAL_BIND_HOSTS:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_LOCAL_HOSTS))
    app.state.mgr = mgr
    app.state.begin_shutdown = begin_shutdown
    app.state.settings = settings

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse({"error": exc.code, "detail": exc.detail}, status_code=exc.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": f"http_{exc.status_code}", "detail": str(exc.detail)},
                            status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return JSONResponse({"error": "validation_error", "detail": detail}, status_code=400)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"error": "internal_error", "detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    # --- meta ------------------------------------------------------------------------
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/providers")
    async def list_providers():
        return [p.info() for p in providers.values()]

    @app.get("/api/budget")
    async def get_budget():
        return {**budget.snapshot(), "qpu_estimate": {
            "base_seconds": ESTIMATE_BASE_SECONDS, "seconds_per_shot": ESTIMATE_SECONDS_PER_SHOT,
            "formula": ESTIMATE_FORMULA, "is_estimate": True}}

    @app.get("/api/jobs")
    async def list_jobs():
        return runner.list()

    @app.get("/api/events")
    async def global_events():
        sub = bus.subscribe(None)
        return _sse(sub, None)

    def _sse(sub, first, on_close=None):
        return StreamingResponse(
            sse_stream(sub, bus, first=first, on_close=on_close),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- pools -----------------------------------------------------------------------
    @app.get("/api/pools")
    async def list_pools():
        return [p.summary() for p in await asyncio.to_thread(pools.list)]

    @app.post("/api/pools/prefetch", status_code=202)
    async def prefetch(body: Prefetch):
        p = await feeder.prefetch(body.provider, body.num_qubits, body.shots)
        return {"job": p.record.to_dict(), "pool_key": p.pool_key}

    @app.delete("/api/pools/{key}")
    async def delete_pool(key: str):
        if not _POOL_KEY.match(key):
            raise AppError(400, "invalid_pool_key", f"pool key must look like provider__implementation__qubits, got {key!r}")
        await asyncio.to_thread(pools.delete, key)  # takes the pool lock; may wait on another process
        return {"deleted": key}

    # --- sessions --------------------------------------------------------------------
    @app.post("/api/sessions", status_code=201)
    async def create_session(body: SessionCreate):
        return (await mgr.create(body.model_dump())).to_dict()

    @app.get("/api/sessions/{sid}")
    async def get_session(sid: str):
        return mgr.get(sid).to_dict()

    @app.patch("/api/sessions/{sid}")
    async def patch_session(sid: str, body: SessionPatch):
        return mgr.switch_mode(mgr.get(sid), body.mode).to_dict()

    @app.post("/api/sessions/{sid}/stream/start")
    async def stream_start(sid: str):
        s = mgr.get(sid)
        await mgr.start_stream(s)
        return s.to_dict()

    @app.post("/api/sessions/{sid}/stream/stop")
    async def stream_stop(sid: str):
        s = mgr.get(sid)
        await mgr.stop_stream(s)
        return s.to_dict()

    @app.post("/api/sessions/{sid}/press")
    async def press(sid: str):
        return await mgr.press(mgr.get(sid))

    @app.post("/api/sessions/{sid}/predict")
    async def predict(sid: str, body: Guess):
        return await mgr.predict(mgr.get(sid), body.guess)

    @app.post("/api/sessions/{sid}/perceive/prepare")
    async def perceive_prepare(sid: str):
        return await mgr.perceive_prepare(mgr.get(sid))

    @app.post("/api/sessions/{sid}/perceive/reveal")
    async def perceive_reveal(sid: str, body: Reveal):
        return await mgr.perceive_reveal(mgr.get(sid), body.trial_id, body.guess)

    @app.get("/api/sessions/{sid}/events")
    async def session_events(sid: str):
        s = mgr.get(sid)
        sub = bus.subscribe(sid)
        # When the last listener of a session goes away (tab closed, page reloaded) its stream
        # stops: nothing in the UI could stop it any more, and on ibm every cycle is a QPU job.
        return _sse(sub, ("stats", {"stats": s.stats_dict(), "session_id": sid}),
                    on_close=lambda: mgr.listener_gone(s))

    app.mount("/", _RevalidatedStaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
