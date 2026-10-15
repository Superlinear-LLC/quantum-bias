"""Failures surface; nothing silently substitutes other bits."""
import pytest
from fastapi.testclient import TestClient

from conftest import make_settings, new_session, wait_for
from fakes import FakeIbm
from quantum_lite.app import create_app
from quantum_lite.providers.ibm import IbmProvider
from quantum_lite.providers.sim import SimProvider

from test_ibm_provider import TOKEN, app_for


def test_ibm_job_error_surfaces_and_no_sim_bits(tmp_path):
    fake = FakeIbm(statuses=("QUEUED", "ERROR"), error="Error code 1517: backend offline")
    with TestClient(app_for(tmp_path, fake)) as c:
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=10)
        for path, body in (("press", None), ("predict", {"guess": 1}), ("perceive/prepare", None)):
            r = c.post(f"/api/sessions/{s['id']}/{path}", json=body)
            assert r.status_code == 502, r.text
            assert r.json()["error"] == "provider_error" and "backend offline" in r.json()["detail"]
        d = c.get(f"/api/sessions/{s['id']}").json()
        assert d["stats"]["total"]["shots"] == 0 and d["perceive_pending"] is False
        jobs = c.get("/api/jobs").json()
        assert jobs and {j["provider"] for j in jobs} == {"ibm"} and all(j["state"] == "error" for j in jobs)
        assert "backend offline" in jobs[0]["error"]


def test_ibm_stream_error_is_reported_and_stops(tmp_path):
    fake = FakeIbm(statuses=("ERROR",), error="quota exhausted")
    with TestClient(app_for(tmp_path, fake)) as c:
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=10)
        c.post(f"/api/sessions/{s['id']}/stream/start")
        wait_for(lambda: not c.get(f"/api/sessions/{s['id']}").json()["stream"]["running"], what="stream stop")
        d = c.get(f"/api/sessions/{s['id']}").json()
        assert "quota exhausted" in d["stream"]["error"] and d["stats"]["total"]["shots"] == 0


def test_service_construction_error_propagates_on_session_create(tmp_path):
    fake = FakeIbm()

    def broken(**kw):
        raise RuntimeError("401 Unauthorized: invalid API key")

    settings = make_settings(tmp_path, ibm_token=TOKEN)
    prov = IbmProvider(settings, service_factory=broken, sampler_cls=fake.sampler_cls)
    with TestClient(create_app(settings, providers={"sim": SimProvider(settings), "ibm": prov})) as c:
        r = c.post("/api/sessions", json={"provider": "ibm", "mode": "live",
                                               "stream_seconds": 30, "stream_max_jobs": 2})
        assert r.status_code == 502 and "401 Unauthorized" in r.json()["detail"]
        assert TOKEN not in r.text


def test_failed_ibm_prefetch_adds_nothing_and_reports(tmp_path):
    fake = FakeIbm(statuses=("ERROR",), error="rejected")
    with TestClient(app_for(tmp_path, fake)) as c:
        r = c.post("/api/pools/prefetch", json={"provider": "ibm", "num_qubits": 3, "shots": 50})
        assert r.status_code == 202
        job_id = r.json()["job"]["id"]
        wait_for(lambda: next(j for j in c.get("/api/jobs").json() if j["id"] == job_id)["state"] == "error",
                 what="job error")
        assert c.get("/api/pools").json() == []


def test_empty_pool_does_not_go_live_ibm(tmp_path):
    fake = FakeIbm()
    with TestClient(app_for(tmp_path, fake)) as c:
        s = new_session(c, provider="ibm", mode="buffered", num_qubits=3)
        r = c.post(f"/api/sessions/{s['id']}/press")
        assert r.status_code == 409 and r.json()["error"] == "pool_empty"
        assert fake.jobs == 0 and c.get("/api/jobs").json() == []


def test_unconfigured_ibm_is_not_replaced_by_sim(client):
    r = client.post("/api/sessions", json={"provider": "ibm", "mode": "buffered"})
    assert r.status_code == 409 and r.json()["error"] == "provider_unavailable"
    r = client.post("/api/pools/prefetch", json={"provider": "ibm", "num_qubits": 3, "shots": 10})
    assert r.status_code == 409
    assert client.get("/api/pools").json() == [] and client.get("/api/jobs").json() == []
