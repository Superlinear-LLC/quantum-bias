"""quantum-lite: serve (default), prefetch, pools."""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from quantum_lite.budget import IbmBudget
from quantum_lite.config import DEFAULT_HOST, DEFAULT_PORT, Settings, load_settings
from quantum_lite.errors import AppError, ConfigError
from quantum_lite.jobs import estimate_qpu_seconds
from quantum_lite.pool_feed import bank, check_prefetch
from quantum_lite.pools import PoolStore, pool_key

LOCAL_HOSTS = {"127.0.0.1", "localhost"}  # not ::1: see app.LOCAL_BIND_HOSTS
COMMANDS = {"serve", "prefetch", "pools"}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantum-lite", description="Quantum Bias Lab")
    p.add_argument("--env-file", type=Path, default=None, help="path to a .env file (default: .env in the project root, if present)")
    sub = p.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="run the web app (default)")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--open", action="store_true", help="open the browser")
    pre = sub.add_parser("prefetch", help="fill a pool on disk, then exit")
    pre.add_argument("--provider", default="sim")
    pre.add_argument("--qubits", type=int, default=5)
    pre.add_argument("--shots", type=int, default=20000)
    sub.add_parser("pools", help="list pools on disk")
    return p


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    # `quantum-lite --port 8781` means `serve`: insert it unless a command is already named.
    # A bare -h/--help alone is the top-level help; with other flags it is serve's.
    if not any(a in COMMANDS for a in args_in) and not (args_in and set(args_in) <= {"-h", "--help"}):
        args_in = _split_global(args_in)
    args = _parser().parse_args(args_in)
    try:
        settings = load_settings(args.env_file)
        if args.command == "prefetch":
            return _prefetch(settings, args)
        if args.command == "pools":
            return _pools(settings)
        return _serve(settings, args)
    except (AppError, ConfigError) as exc:
        # Expected, user-facing failures: the reason on one line. Anything else propagates.
        code = exc.code if isinstance(exc, AppError) else "config_error"
        detail = exc.detail if isinstance(exc, AppError) else str(exc)
        print(f"error: {code}: {detail}", file=sys.stderr)
        return 1


def _split_global(argv: list[str]) -> list[str]:
    """Move --env-file (global) in front of `serve`; everything else belongs to serve."""
    head: list[str] = []
    rest: list[str] = []
    it = iter(argv)
    for a in it:
        if a == "--env-file":
            head += [a, next(it, "")]
        elif a.startswith("--env-file="):
            head.append(a)
        else:
            rest.append(a)
    return head + ["serve"] + rest


GRACEFUL_SHUTDOWN_SECONDS = 5


def make_server(app, host: str, port: int, open_url: str | None = None):
    """A uvicorn Server that (a) opens the browser only once it is listening, and (b) stops the
    streams and ends the SSE responses *before* waiting for connections to close.

    Stock uvicorn waits for every connection first, and an SSE response never ends by itself,
    so Ctrl+C with a UI tab open hung while running streams kept consuming pools or submitting
    jobs. Remaining connections are cancelled after GRACEFUL_SHUTDOWN_SECONDS.
    """
    import uvicorn

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)  # exits the process if the bind fails: no browser then
            if open_url and self.started:
                import webbrowser

                webbrowser.open(open_url)

        async def shutdown(self, sockets=None):
            await app.state.begin_shutdown()
            await super().shutdown(sockets)

    config = uvicorn.Config(app, host=host, port=port, log_level="info",
                            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS)
    return Server(config)


def _serve(settings: Settings, args) -> int:
    from quantum_lite.app import create_app

    if args.host not in LOCAL_HOSTS:
        print(
            f"WARNING: binding {args.host}. This app has no authentication; anyone who can "
            "reach this address can use your session (and your IBM budget).",
            file=sys.stderr,
        )
    url_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    if ":" in url_host:
        url_host = f"[{url_host}]"  # an IPv6 literal needs brackets in a URL
    url = f"http://{url_host}:{args.port}/"
    app = create_app(settings, bind_host=args.host)
    make_server(app, args.host, args.port, open_url=url if args.open else None).run()
    return 0


def _prefetch(settings: Settings, args) -> int:
    from quantum_lite.app import build_providers

    providers = build_providers(settings)
    budget = IbmBudget(settings)
    provider = check_prefetch(providers, args.provider, args.qubits, args.shots, budget)
    if provider.id == "ibm":
        budget.reserve(args.shots)  # before resolve: a ceiling never prints device:, never contacts IBM
    implementation = provider.resolve()
    print(f"device: {implementation}", flush=True)
    if provider.id == "ibm" and not settings.ibm_backend:
        print(f"note: IBM_BACKEND is unset; the pool is for {implementation}. Set IBM_BACKEND={implementation} "
              "when you start the app, or it may pick another device and find this pool empty.", flush=True)
    pools = PoolStore(settings.data_dir)

    def on_state(state, *, queue_position=None, job_id=None, backend=None):
        print(f"job {job_id}: {state} on {backend}", flush=True)

    m = provider.run(args.qubits, args.shots, on_state, threading.Event())
    pool = bank(pools, provider.id, args.qubits, m, expect=implementation)
    est = f"{estimate_qpu_seconds(args.shots):.1f} (estimate: 2 + 0.00035 * shots)" if provider.id == "ibm" \
        else "0 (simulator)"
    print(f"job_id: {m.job_id}")
    print(f"shots: {args.shots}")
    print(f"estimated QPU seconds: {est}")
    print(f"pool: {pool_key(provider.id, implementation, args.qubits)} now holds {pool.available} unconsumed shots")
    return 0


def _pools(settings: Settings) -> int:
    pools = PoolStore(settings.data_dir).list()
    if not pools:
        print(f"no pools in {settings.data_dir / 'pools'}")
        return 0
    for p in pools:
        print(f"{p.key}\tavailable={p.available}\ttotal={p.total}\tconsumed={p.cursor}\tblocks={len(p.blocks)}")
    return 0
