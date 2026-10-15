import pytest

from conftest import make_settings, new_session, prefetch_and_wait, wait_for
from fastapi.testclient import TestClient
from quantum_lite.app import create_app


@pytest.fixture
def pooled(client):
    prefetch_and_wait(client, n=5, shots=2000)
    return client


def test_health_static_and_providers(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert "Quantum Bias Lab" in client.get("/").text
    providers = {p["id"]: p for p in client.get("/api/providers").json()}
    assert providers["sim"] == {"id": "sim", "variant": "simulator", "implementation": "aer_ideal",
                                "configured": True, "available": True, "reason": None}
    assert providers["ibm"]["configured"] is False
    assert providers["ibm"]["reason"] == "IBM_QUANTUM_TOKEN not set"
    assert providers["ibm"]["implementation"] is None


def test_error_body_shape(client):
    r = client.get("/api/sessions/nope")
    assert r.status_code == 404 and r.json()["error"] == "session_not_found" and "nope" in r.json()["detail"]
    r = client.post("/api/sessions", json={"num_qubits": "x"})
    assert r.status_code == 400 and r.json()["error"] == "validation_error"


def test_session_validation(client):
    r = client.post("/api/sessions", json={"provider": "sim", "mode": "live", "num_qubits": 4, "aggregation": "majority"})
    assert r.status_code == 400 and r.json()["error"] == "invalid_aggregation"
    assert client.post("/api/sessions", json={"num_qubits": 17}).status_code == 400
    assert client.post("/api/sessions", json={"num_qubits": 0}).status_code == 400
    assert client.post("/api/sessions", json={"mode": "turbo"}).status_code == 400
    r = client.post("/api/sessions", json={"provider": "ibm", "mode": "live"})
    assert r.status_code == 409 and "IBM_QUANTUM_TOKEN not set" in r.json()["detail"]
    assert client.post("/api/sessions", json={"provider": "nope"}).status_code == 400
    r = client.post("/api/sessions", json={"auto_refill": True, "stream_seconds": 60})
    assert r.status_code == 400 and r.json()["error"] == "invalid_refill"


@pytest.mark.parametrize("mode", ["live", "buffered"])
def test_press(pooled, mode):
    s = new_session(pooled, mode=mode)
    r = pooled.post(f"/api/sessions/{s['id']}/press")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bit"] in (0, 1)
    prov = body["provenance"]
    assert prov["source"] == mode and prov["provider"] == "sim" and prov["backend"] == "aer_ideal"
    assert prov["job_id"] and prov["measured_at"]
    seg = body["stats"]["segments"][0]
    assert body["stats"]["total"]["shots"] == 1 and seg["stats"]["raw"]["n"] == 5
    assert seg["stats"]["aggregated"]["n"] == 1


@pytest.mark.parametrize("mode", ["live", "buffered"])
def test_predict(pooled, mode):
    s = new_session(pooled, mode=mode)
    r = pooled.post(f"/api/sessions/{s['id']}/predict", json={"guess": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hit"] == (body["bit"] == 1)
    assert body["measured_before_guess"] is (mode == "buffered")
    assert body["provenance"]["source"] == mode
    p = body["stats"]["total"]["predict"]
    assert p["trials"] == 1 and p["hits"] == int(body["hit"])
    assert pooled.post(f"/api/sessions/{s['id']}/predict", json={"guess": 2}).status_code == 400


@pytest.mark.parametrize("mode", ["live", "buffered"])
def test_perceive(pooled, mode):
    s = new_session(pooled, mode=mode)
    sid = s["id"]
    r = pooled.post(f"/api/sessions/{sid}/perceive/prepare")
    assert r.status_code == 200, r.text
    prep = r.json()
    assert set(prep) == {"trial_id", "source"} and prep["source"] == mode
    # hidden bit is not in stats yet
    assert pooled.get(f"/api/sessions/{sid}").json()["stats"]["total"]["shots"] == 0
    # second prepare while pending
    r2 = pooled.post(f"/api/sessions/{sid}/perceive/prepare")
    assert r2.status_code == 409 and r2.json()["error"] == "perceive_pending"
    # wrong trial id
    assert pooled.post(f"/api/sessions/{sid}/perceive/reveal",
                       json={"trial_id": "bogus", "guess": 0}).status_code == 404
    r = pooled.post(f"/api/sessions/{sid}/perceive/reveal", json={"trial_id": prep["trial_id"], "guess": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hit"] == (body["bit"] == 0) and body["provenance"]["source"] == mode
    assert body["stats"]["total"]["perceive"]["trials"] == 1
    assert body["stats"]["total"]["shots"] == 1
    # reveal consumed it; a new prepare is allowed
    assert pooled.post(f"/api/sessions/{sid}/perceive/prepare").status_code == 200
    # reveal twice is refused
    assert pooled.post(f"/api/sessions/{sid}/perceive/reveal",
                       json={"trial_id": prep["trial_id"], "guess": 1}).status_code == 404


@pytest.mark.parametrize("mode", ["live", "buffered"])
def test_stream_start_events_stop(pooled, mode):
    s = new_session(pooled, mode=mode, shots_per_job=10, interval_ms=20)
    sid = s["id"]
    r = pooled.post(f"/api/sessions/{sid}/stream/start")
    assert r.status_code == 200 and r.json()["stream"]["running"] is True
    assert pooled.post(f"/api/sessions/{sid}/stream/start").status_code == 409
    wait_for(lambda: pooled.get(f"/api/sessions/{sid}").json()["stats"]["total"]["shots"] >= 30, what="stream shots")
    r = pooled.post(f"/api/sessions/{sid}/stream/stop")
    assert r.status_code == 200 and r.json()["stream"]["running"] is False
    assert pooled.post(f"/api/sessions/{sid}/stream/stop").status_code == 409
    total = pooled.get(f"/api/sessions/{sid}").json()["stats"]["total"]
    assert total["shots"] % 10 == 0
    assert total["raw"]["n"] == total["shots"] * 5 and total["aggregated"]["n"] == total["shots"]
    after = pooled.get(f"/api/sessions/{sid}").json()["stats"]["total"]["shots"]
    assert after == total["shots"]  # really stopped


def test_aggregation_xor_and_majority_in_stats(pooled):
    for agg in ("xor", "majority"):
        s = new_session(pooled, mode="live", aggregation=agg)
        body = pooled.post(f"/api/sessions/{s['id']}/press").json()
        assert body["bit"] in (0, 1) and body["stats"]["total"]["raw"]["n"] == 5


def test_mode_switch_creates_segment_and_keeps_stats_apart(pooled):
    s = new_session(pooled, mode="buffered")
    sid = s["id"]
    pooled.post(f"/api/sessions/{sid}/press")
    r = pooled.patch(f"/api/sessions/{sid}", json={"mode": "live"})
    assert r.status_code == 200 and r.json()["mode"] == "live"
    assert pooled.patch(f"/api/sessions/{sid}", json={"mode": "live"}).status_code == 400
    body = pooled.post(f"/api/sessions/{sid}/press").json()
    assert body["provenance"]["source"] == "live" and body["provenance"]["segment"] == 1
    segs = body["stats"]["segments"]
    assert [x["mode"] for x in segs] == ["buffered", "live"]
    assert [x["stats"]["shots"] for x in segs] == [1, 1]
    assert segs[0]["ended_at"] is not None and segs[1]["ended_at"] is None
    assert body["stats"]["total"]["shots"] == 2


def test_empty_pool_is_409(client):
    s = new_session(client, mode="buffered")
    r = client.post(f"/api/sessions/{s['id']}/press")
    assert r.status_code == 409 and r.json()["error"] == "pool_empty"
    assert s["pool_key"] in r.json()["detail"]
    r = client.post(f"/api/sessions/{s['id']}/stream/start")
    assert r.status_code == 409 and r.json()["error"] == "pool_empty"
    assert client.get("/api/jobs").json() == []  # nothing went live behind our back


def test_pool_exhausted_mid_stream_stops_with_error(client):
    prefetch_and_wait(client, n=5, shots=25)
    s = new_session(client, mode="buffered", shots_per_job=10, interval_ms=10)
    sid = s["id"]
    client.post(f"/api/sessions/{sid}/stream/start")
    wait_for(lambda: not client.get(f"/api/sessions/{sid}").json()["stream"]["running"], what="stream to stop")
    d = client.get(f"/api/sessions/{sid}").json()
    assert d["stream"]["error"].startswith("pool_empty")
    assert d["stats"]["total"]["shots"] == 20  # two full cycles, never padded


def test_prefetch_pools_and_delete(client):
    key = prefetch_and_wait(client, n=3, shots=500)
    pools = client.get("/api/pools").json()
    assert [p["key"] for p in pools] == [key] and pools[0]["available_shots"] == 500
    jobs = client.get("/api/jobs").json()
    assert jobs[0]["kind"] == "prefetch" and jobs[0]["provider"] == "sim" and jobs[0]["job_id"].startswith("sim-")
    assert jobs[0]["estimated_qpu_seconds"] is None
    assert client.delete(f"/api/pools/{key}").status_code == 200
    assert client.get("/api/pools").json() == []
    assert client.delete(f"/api/pools/{key}").status_code == 404


def test_pool_is_consumed_and_persisted_across_restart(settings):
    with TestClient(create_app(settings)) as c:
        key = prefetch_and_wait(c, n=5, shots=100)
        s = new_session(c, mode="buffered")
        for _ in range(3):
            c.post(f"/api/sessions/{s['id']}/press")
        assert c.get("/api/pools").json()[0]["available_shots"] == 97
    with TestClient(create_app(settings)) as c2:
        assert c2.get("/api/pools").json()[0]["available_shots"] == 97


def test_sim_latency_in_job_records(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, sim_latency_seconds=0.2))) as c:
        s = new_session(c, mode="live")
        c.post(f"/api/sessions/{s['id']}/press")
        job = c.get("/api/jobs").json()[0]
        assert job["simulated_latency"] is True and job["state"] == "done" and job["kind"] == "press"
        assert job["shots"] == 1 and job["finished_at"]


def test_budget_endpoint_for_sim_only_run(client):
    b = client.get("/api/budget").json()
    assert b["shots"] == {"used": 0, "ceiling": 200000} and b["jobs"] == {"used": 0, "ceiling": 100}
