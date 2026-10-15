"""Shutdown, browser open, mode guards, discarded bits, registry bounds, UI files."""
import socket
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import make_settings, new_session, prefetch_and_wait, wait_for
from fakes import FakeIbm
from quantum_lite import cli
from quantum_lite.app import create_app
from test_ibm_provider import app_for



def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Running:
    """A real uvicorn server (cli.make_server) in a thread."""

    def __init__(self, app, open_url=None):
        self.port = free_port()
        self.server = cli.make_server(app, "127.0.0.1", self.port, open_url=open_url)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        wait_for(lambda: self.server.started, what="server start")

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self, within):
        t0 = time.monotonic()
        self.server.should_exit = True  # what the SIGINT handler sets
        self.thread.join(within)
        took = time.monotonic() - t0
        assert not self.thread.is_alive(), f"server still running {within}s after the stop signal"
        return took


@pytest.fixture
def allow_127(monkeypatch):
    return None  # 127.0.0.1 is always allowed; kept for readability


def test_shutdown_with_open_sse_client_stops_streams_and_exits_fast(tmp_path):
    app = create_app(make_settings(tmp_path, sim_latency_seconds=0.05))
    srv = Running(app)
    seen = []
    got_batch = threading.Event()

    def listen(sid):
        with httpx.stream("GET", f"{srv.base}/api/sessions/{sid}/events", timeout=30) as r:
            for line in r.iter_lines():
                seen.append(line)
                if line.startswith("event: batch"):
                    got_batch.set()

    s = httpx.post(f"{srv.base}/api/sessions", json={"provider": "sim", "mode": "live", "num_qubits": 3,
                                                      "shots_per_job": 5, "stream_seconds": 60}).json()
    t = threading.Thread(target=listen, args=(s["id"],), daemon=True)
    t.start()
    httpx.post(f"{srv.base}/api/sessions/{s['id']}/stream/start").raise_for_status()
    assert got_batch.wait(10)
    took = srv.stop(within=4)  # stock uvicorn waits for the SSE connection forever
    assert took < 4
    t.join(5)
    assert not t.is_alive()  # the client saw the response end
    runner = app.state.mgr.runner
    n_jobs = len(runner.records)
    time.sleep(0.3)
    assert len(runner.records) == n_jobs  # no stream cycle after the signal
    assert not app.state.mgr.get(s["id"]).streaming


def test_browser_opens_only_after_the_port_is_listening(tmp_path, monkeypatch):
    import webbrowser

    port_holder = {}
    result = {}

    def fake_open(url):
        try:
            with socket.create_connection(("127.0.0.1", port_holder["port"]), timeout=2):
                result["connected"] = True
        except OSError as exc:
            result["connected"] = f"{type(exc).__name__}: {exc}"
        return True

    monkeypatch.setattr(webbrowser, "open", fake_open)
    app = create_app(make_settings(tmp_path))
    port = free_port()
    port_holder["port"] = port
    server = cli.make_server(app, "127.0.0.1", port, open_url=f"http://127.0.0.1:{port}/")
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    wait_for(lambda: "connected" in result, what="browser open")
    server.should_exit = True
    th.join(5)
    assert result["connected"] is True, result


def test_browser_is_not_opened_when_the_bind_fails(tmp_path, monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        server = cli.make_server(create_app(make_settings(tmp_path)), "127.0.0.1", port,
                                 open_url=f"http://127.0.0.1:{port}/")
        with pytest.raises(SystemExit):
            server.run()
    assert opened == []


def test_docs_pages_that_need_a_cdn_are_off_but_the_schema_is_served(client):
    assert client.get("/docs").status_code == 404 and client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 200


def test_delete_pool_rejects_keys_that_are_not_pool_keys(client):
    r = client.delete("/api/pools/a%5Cb")
    assert r.status_code == 400 and r.json()["error"] == "invalid_pool_key"
    r = client.delete("/api/pools/sim__aer_ideal__5")
    assert r.status_code == 404


def test_buffered_stream_starts_while_an_abandoned_live_job_is_still_running(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, sim_latency_seconds=0.6))) as c:
        prefetch_and_wait(c, n=5, shots=500)
        s = new_session(c, mode="live", shots_per_job=10)
        sid = s["id"]
        assert c.post(f"/api/sessions/{sid}/stream/start").status_code == 200
        assert c.post(f"/api/sessions/{sid}/stream/stop").status_code == 200
        assert c.patch(f"/api/sessions/{sid}", json={"mode": "buffered"}).status_code == 200
        r = c.post(f"/api/sessions/{sid}/stream/start")  # a buffered stream submits no job
        assert r.status_code == 200, r.text
        c.post(f"/api/sessions/{sid}/stream/stop")


def test_stopped_live_stream_job_bits_go_to_the_pool_not_the_void(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, sim_latency_seconds=0.4))) as c:
        s = new_session(c, mode="live", shots_per_job=50)
        sid = s["id"]
        c.post(f"/api/sessions/{sid}/stream/start")
        time.sleep(0.1)
        c.post(f"/api/sessions/{sid}/stream/stop")
        job = wait_for(lambda: next((j for j in c.get("/api/jobs").json()
                                     if j["kind"] == "stream" and j["pooled_to"]), None), what="pooled job")
        assert job["discarded"] is True and job["pooled_to"] == s["pool_key"]
        pool = next(p for p in c.get("/api/pools").json() if p["key"] == s["pool_key"])
        assert pool["available_shots"] == 50 and pool["blocks"][0]["job_id"] == job["job_id"]
        total = c.get(f"/api/sessions/{sid}").json()["stats"]["total"]
        assert total["raw"]["n"] == 0  # never counted as live bits


def test_finished_sim_records_are_bounded_but_ibm_records_are_all_kept(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        c.app.state.mgr.runner.max_finished_sim_records = 3
        s = new_session(c, mode="live", shots_per_job=1)
        for _ in range(8):
            assert c.post(f"/api/sessions/{s['id']}/press").status_code == 200
        assert len(c.get("/api/jobs").json()) == 3
    with TestClient(app_for(tmp_path / "ibm", FakeIbm())) as c:
        c.app.state.mgr.runner.max_finished_sim_records = 1
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=1)
        for _ in range(4):
            assert c.post(f"/api/sessions/{s['id']}/press").status_code == 200
        assert len(c.get("/api/jobs").json()) == 4
