from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from quantum_lite import config as config_mod
from quantum_lite.app import create_app
from quantum_lite.config import Settings

# The user's real .env path, captured before any test patches it. Only test_live_ibm.py reads it.
REAL_ENV_FILE = config_mod.DEFAULT_ENV_FILE
_CONFIG_PREFIXES = ("IBM_", "SIM_", "QL_")
_CONFIG_NAMES = ("QISKIT_IBM_TOKEN",)


@pytest.fixture(autouse=True)
def _isolated_config(request, monkeypatch, tmp_path_factory):
    """No test may see the user's .env or configuration variables.

    With a token there, a CLI test would reach the real IbmProvider and submit a real QPU job.
    test_live_ibm.py is the one module that reads the real configuration (it skips unless
    QL_LIVE_IBM=1).
    """
    if request.module.__name__ == "test_live_ibm":
        return
    monkeypatch.setattr(config_mod, "DEFAULT_ENV_FILE", tmp_path_factory.mktemp("noenv") / "absent.env")
    import os

    for name in list(os.environ):
        if name.startswith(_CONFIG_PREFIXES) or name in _CONFIG_NAMES:
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _testserver_host(monkeypatch):
    """TestClient sends Host: testserver; the app accepts only local host names."""
    from quantum_lite import app as app_mod

    monkeypatch.setattr(app_mod, "ALLOWED_LOCAL_HOSTS", (*app_mod.ALLOWED_LOCAL_HOSTS, "testserver"))


def make_settings(tmp_path, **kw) -> Settings:
    return Settings(data_dir=tmp_path / "data", **kw)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def ibm_app(tmp_path, fake, **kw):
    """An app whose ibm provider talks to the FakeIbm boundary (no network, no real token)."""
    from quantum_lite.providers.ibm import IbmProvider
    from quantum_lite.providers.sim import SimProvider

    settings = make_settings(tmp_path, ibm_token="SECRET-TOKEN-0123456789", **kw)
    prov = IbmProvider(settings, service_factory=fake.service_factory, sampler_cls=fake.sampler_cls, poll_seconds=0.01)
    return create_app(settings, providers={"sim": SimProvider(settings), "ibm": prov})


def wait_for(predicate, timeout=20.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


async def until(predicate, what, timeout=10.0):
    """Await `predicate` on the running loop; fail (never hang) after `timeout` s."""
    import asyncio

    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


def prefetch_and_wait(client, provider="sim", n=5, shots=2000):
    def available():
        return sum(p["available_shots"] for p in client.get("/api/pools").json()
                   if p["provider"] == provider and p["n_qubits"] == n)

    before = available()
    r = client.post("/api/pools/prefetch", json={"provider": provider, "num_qubits": n, "shots": shots})
    assert r.status_code == 202, r.text
    job_id = r.json()["job"]["id"]

    def finished():
        job = next(j for j in client.get("/api/jobs").json() if j["id"] == job_id)
        return job if job["state"] in ("done", "error") else None

    job = wait_for(finished, what="prefetch job")
    assert job["state"] == "done", job
    # The job is "done" before its bits are written to the pool (JobRunner.execute, then PoolStore.append).
    wait_for(lambda: available() >= before + shots, what="the prefetched shots in the pool")
    return r.json()["pool_key"]


def new_session(client, **kw):
    body = {"provider": "sim", "mode": "buffered", "num_qubits": 5, "aggregation": "none",
            "shots_per_job": 10, "interval_ms": 20, "stream_seconds": 60}
    body.update(kw)
    if body["provider"] == "ibm":
        body.setdefault("stream_max_jobs", 5)
    r = client.post("/api/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()
