"""IBM provider against a fake service/sampler/job boundary (no network, no token)."""
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from conftest import make_settings, new_session, wait_for
from fakes import FakeIbm
from quantum_lite.app import create_app
from quantum_lite.circuit import lowest_error_qubits
from quantum_lite.errors import ProviderError
from quantum_lite.providers.ibm import IbmProvider, map_status
from quantum_lite.providers.sim import SimProvider

TOKEN = "SECRET-TOKEN-0123456789"


def provider_for(tmp_path, fake, **settings_kw):
    settings = make_settings(tmp_path, ibm_token=TOKEN, **settings_kw)
    prov = IbmProvider(settings, service_factory=fake.service_factory, sampler_cls=fake.sampler_cls,
                       poll_seconds=0.01)
    return settings, prov


def app_for(tmp_path, fake, **settings_kw):
    settings, prov = provider_for(tmp_path, fake, **settings_kw)
    return create_app(settings, providers={"sim": SimProvider(settings), "ibm": prov})


def run_collect(prov, n=3, shots=50):
    states = []
    m = prov.run(n, shots, lambda state, **kw: states.append((state, kw)), threading.Event())
    return m, states


def test_token_passed_to_constructor_and_save_account_never_called(tmp_path, monkeypatch):
    import qiskit_ibm_runtime

    def forbidden(*a, **k):
        raise AssertionError("save_account must never be called")

    monkeypatch.setattr(qiskit_ibm_runtime.QiskitRuntimeService, "save_account", forbidden)
    fake = FakeIbm()
    _, prov = provider_for(tmp_path, fake, ibm_instance="crn:v1:test", ibm_channel="ibm_cloud")
    run_collect(prov)
    assert fake.service_calls == [{"channel": "ibm_cloud", "token": TOKEN, "instance": "crn:v1:test"}]


def test_least_busy_used_when_backend_unset_and_resolved_once(tmp_path):
    fake = FakeIbm()
    _, prov = provider_for(tmp_path, fake)
    run_collect(prov)
    run_collect(prov)
    assert fake.least_busy_calls == [{"operational": True, "simulator": False}]
    assert fake.backend_calls == [] and len(fake.service_calls) == 1


def test_fixed_backend_when_configured(tmp_path):
    fake = FakeIbm()
    _, prov = provider_for(tmp_path, fake, ibm_backend="ibm_torino")
    run_collect(prov)
    assert fake.backend_calls == ["ibm_torino"] and fake.least_busy_calls == []


def test_result_bits_provenance_and_job_mode(tmp_path):
    fake = FakeIbm()
    _, prov = provider_for(tmp_path, fake)
    m, states = run_collect(prov, n=4, shots=64)
    assert m.bits.shape == (64, 4) and m.provider == "ibm" and m.backend == fake.backend.name
    assert m.job_id == "fakejob-1" and m.measured_at
    assert fake.sampler_runs == [{"mode": fake.backend, "shots": 64}]
    assert list(m.physical_qubits) == lowest_error_qubits(fake.backend.target, 4)


def test_states_are_mapped_and_queue_position_is_null(tmp_path):
    # Only statuses a real RuntimeJobV2.status() returns after submission (the API's own values).
    fake = FakeIbm(statuses=("QUEUED", "QUEUED", "RUNNING", "DONE"))
    _, prov = provider_for(tmp_path, fake)
    _, states = run_collect(prov)
    assert [s for s, _ in states] == ["queued", "running"]
    assert all(kw["queue_position"] is None for _, kw in states)
    assert all(kw["job_id"] == "fakejob-1" for _, kw in states)


def test_first_reported_state_is_what_ibm_said_not_an_assumed_queued(tmp_path):
    fake = FakeIbm(statuses=("RUNNING", "DONE"))
    _, prov = provider_for(tmp_path, fake)
    _, states = run_collect(prov)
    assert [s for s, _ in states] == ["running"]  # never 'queued': IBM never said so


def test_unmapped_raw_status_names_the_job(tmp_path):
    # RuntimeJobV2._status_from_job_response passes an unmapped API status through unchanged.
    fake = FakeIbm(statuses=("QUEUED", "VALIDATING"))
    _, prov = provider_for(tmp_path, fake)
    with pytest.raises(ProviderError, match=r"IBM job fakejob-1: unrecognised IBM job status: 'VALIDATING'"):
        run_collect(prov)


def test_map_status_table_and_unknown():
    assert [map_status(s) for s in ("INITIALIZING", "QUEUED", "RUNNING", "DONE", "ERROR", "CANCELLED")] == \
        ["initializing", "queued", "running", "done", "error", "cancelled"]
    with pytest.raises(ProviderError, match="unrecognised"):
        map_status("WEIRD")


def test_failed_job_raises_with_real_message(tmp_path):
    fake = FakeIbm(statuses=("QUEUED", "ERROR"), error="Error code 1517: no capacity")
    _, prov = provider_for(tmp_path, fake)
    with pytest.raises(ProviderError, match="no capacity"):
        run_collect(prov)


def test_cancelled_job_reports_that_no_message_exists(tmp_path):
    # A real RuntimeJobV2 returns error_message() None for CANCELLED, whatever the API said.
    fake = FakeIbm(statuses=("QUEUED", "CANCELLED"), error="Error code 1517: no capacity")
    _, prov = provider_for(tmp_path, fake)
    with pytest.raises(ProviderError, match="cancelled: no error message reported"):
        run_collect(prov)


def test_no_ibm_call_at_startup(tmp_path):
    fake = FakeIbm()
    with TestClient(app_for(tmp_path, fake)) as c:
        c.get("/api/health")
        providers = {p["id"]: p for p in c.get("/api/providers").json()}
        c.get("/api/budget")
        c.get("/api/pools")
    assert providers["ibm"]["configured"] is True and providers["ibm"]["implementation"] is None
    assert fake.service_calls == [] and fake.jobs == 0


def test_token_never_leaks(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeIbm()
    settings, _ = provider_for(tmp_path, fake)
    with TestClient(app_for(tmp_path, fake)) as c:
        listing = c.get("/api/providers")
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=10)
        r = c.post(f"/api/sessions/{s['id']}/press")
        assert r.status_code == 200, r.text
        blobs = [listing.text, c.get("/api/jobs").text, c.get("/api/budget").text, r.text, str(s),
                 c.get("/").text, repr(settings), str(settings)]
    assert not any(TOKEN in b for b in blobs)
    assert TOKEN not in caplog.text


def test_ibm_session_live_end_to_end_and_budget(tmp_path):
    fake = FakeIbm()
    with TestClient(app_for(tmp_path, fake)) as c:
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=10)
        assert s["implementation"] == fake.backend.name
        assert s["physical_qubits"] == lowest_error_qubits(fake.backend.target, 3)
        body = c.post(f"/api/sessions/{s['id']}/press").json()
        assert body["provenance"]["provider"] == "ibm" and body["provenance"]["source"] == "live"
        assert body["provenance"]["job_id"] == "fakejob-1"
        c.post(f"/api/sessions/{s['id']}/predict", json={"guess": 0})
        job = c.get("/api/jobs").json()[0]
        assert job["provider"] == "ibm" and job["shots"] == 1 and job["state"] == "done"
        assert job["estimated_qpu_seconds"] == pytest.approx(2 + 0.00035)
        assert job["queue_position"] is None
        assert c.get("/api/budget").json()["shots"]["used"] == 2
        assert c.get("/api/budget").json()["jobs"]["used"] == 2


def test_prefetch_ibm_fills_pool_then_buffered_session_uses_it(tmp_path):
    fake = FakeIbm()
    with TestClient(app_for(tmp_path, fake)) as c:
        r = c.post("/api/pools/prefetch", json={"provider": "ibm", "num_qubits": 3, "shots": 200})
        assert r.status_code == 202
        wait_for(lambda: c.get("/api/pools").json(), what="pool")
        s = new_session(c, provider="ibm", mode="buffered", num_qubits=3)
        body = c.post(f"/api/sessions/{s['id']}/press").json()
        assert body["provenance"]["source"] == "buffered" and body["provenance"]["provider"] == "ibm"
        assert body["provenance"]["job_id"] == "fakejob-1"
        assert fake.jobs == 1  # buffered press submitted nothing


def test_per_job_ceiling_is_400(tmp_path):
    fake = FakeIbm()
    with TestClient(app_for(tmp_path, fake, ibm_max_shots_per_job=100)) as c:
        r = c.post("/api/pools/prefetch", json={"provider": "ibm", "num_qubits": 3, "shots": 101})
        assert r.status_code == 400 and r.json()["error"] == "shots_exceed_job_ceiling"
        assert "101" in r.json()["detail"] and "100" in r.json()["detail"]
        r = c.post("/api/sessions", json={"provider": "ibm", "mode": "live", "shots_per_job": 101})
        assert r.status_code == 400
    assert fake.jobs == 0


def test_total_shot_ceiling_is_429_with_numbers(tmp_path):
    fake = FakeIbm()
    with TestClient(app_for(tmp_path, fake, ibm_max_total_shots=3)) as c:
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=10)
        for _ in range(3):
            assert c.post(f"/api/sessions/{s['id']}/press").status_code == 200
        r = c.post(f"/api/sessions/{s['id']}/press")
        assert r.status_code == 429 and r.json()["error"] == "shot_ceiling"
        assert "3" in r.json()["detail"]
        assert fake.jobs == 3


def test_job_ceiling_is_429(tmp_path):
    fake = FakeIbm()
    with TestClient(app_for(tmp_path, fake, ibm_max_jobs=1)) as c:
        s = new_session(c, provider="ibm", mode="live", num_qubits=3, shots_per_job=10)
        assert c.post(f"/api/sessions/{s['id']}/press").status_code == 200
        r = c.post(f"/api/sessions/{s['id']}/press")
        assert r.status_code == 429 and r.json()["error"] == "job_ceiling"


def test_sim_jobs_do_not_touch_ibm_budget(tmp_path):
    fake = FakeIbm()
    with TestClient(app_for(tmp_path, fake, ibm_max_jobs=1)) as c:
        s = new_session(c, provider="sim", mode="live")
        for _ in range(3):
            assert c.post(f"/api/sessions/{s['id']}/press").status_code == 200
        assert c.get("/api/budget").json()["jobs"]["used"] == 0


def test_real_backend_translation_hook_does_not_break_transpile(tmp_path):
    # FakeIbm's backend defines get_translation_stage_plugin() -> 'ibm_dynamic_circuits' like a
    # real IBMBackend; without translation_method='translator' the transpile raises TranspilerError.
    fake = FakeIbm()
    assert fake.backend.get_translation_stage_plugin() == "ibm_dynamic_circuits"
    _, prov = provider_for(tmp_path, fake)
    m, _ = run_collect(prov, n=3, shots=20)
    assert m.bits.shape == (20, 3)


def test_each_state_is_reported_once_and_done_is_left_to_the_runner(tmp_path):
    fake = FakeIbm(statuses=("QUEUED", "RUNNING", "DONE"))
    _, prov = provider_for(tmp_path, fake)
    _, states = run_collect(prov)
    assert [s for s, _ in states] == ["queued", "running"]


def test_cancelled_job_without_message_does_not_print_none(tmp_path):
    fake = FakeIbm(statuses=("QUEUED", "CANCELLED"), error=None)
    _, prov = provider_for(tmp_path, fake)
    with pytest.raises(ProviderError) as exc:
        run_collect(prov)
    assert "None" not in str(exc.value) and "fakejob-1" in str(exc.value) and "cancelled" in str(exc.value)


def test_qubits_without_sx_are_not_chosen():
    from types import SimpleNamespace

    from quantum_lite.circuit import lowest_error_qubits

    errors = {0: 0.001, 1: 0.01, 2: 0.02, 3: 0.03}
    measure = {(q,): SimpleNamespace(error=e) for q, e in errors.items()}

    class Target:
        def __getitem__(self, name):
            assert name == "measure"
            return measure

        def instruction_supported(self, gate, qargs):
            return not (gate == "sx" and qargs == (0,))  # qubit 0 reads best but has no sx

    assert lowest_error_qubits(Target(), 2) == [1, 2]


@pytest.mark.filterwarnings("ignore:The property .*QuantumCircuit.calibrations.*:DeprecationWarning")
def test_real_sampler_v2_path_bit_order_and_layout(tmp_path, monkeypatch):
    # No sampler_cls: the real qiskit_ibm_runtime.SamplerV2 runs in local mode on the fake backend,
    # so ISA validation, PrimitiveResult/BitArray access and the bit reversal are the real ones.
    from quantum_lite.circuit import build_circuit
    from quantum_lite.providers import ibm as ibm_mod

    monkeypatch.setattr(ibm_mod, "build_circuit", lambda n: build_circuit(n, x_on=[0]))
    fake = FakeIbm(statuses=("DONE",))
    settings = make_settings(tmp_path, ibm_token=TOKEN)
    prov = IbmProvider(settings, service_factory=fake.service_factory, poll_seconds=0.01)
    # (qiskit-ibm-runtime's own ISA check reads a pulse property qiskit 1.4 deprecates: filtered above.)
    m, _ = run_collect(prov, n=4, shots=2000)
    assert m.bits.shape == (2000, 4)
    assert m.bits[:, 0].mean() > 0.9  # X on qubit 0, with the fake's readout error
    for col in (1, 2, 3):
        assert abs(m.bits[:, col].mean() - 0.5) < 0.1
    assert list(m.physical_qubits) == lowest_error_qubits(fake.backend.target, 4)


def measured_physical_by_clbit(isa):
    """{clbit index: physical qubit it is measured on} read off a transpiled circuit."""
    out = {}
    for inst in isa.data:
        if inst.operation.name == "measure":
            out[isa.find_bit(inst.clbits[0]).index] = isa.find_bit(inst.qubits[0]).index
    return out


@pytest.mark.parametrize("n", [1, 5, 16])
def test_transpiled_circuit_measures_on_the_reported_physical_qubits(tmp_path, n):
    # Not circular: the reported list is checked against what the submitted ISA circuit does.
    # Without initial_layout the circuit measures qubits 0..n-1 and this fails.
    fake = FakeIbm()
    _, prov = provider_for(tmp_path, fake)
    m, _ = run_collect(prov, n=n, shots=10)
    measured = measured_physical_by_clbit(fake.circuits[0])
    assert measured == {j: q for j, q in enumerate(m.physical_qubits)}
    assert list(m.physical_qubits) != list(range(n)) or n == 1  # the fake's best qubits are not 0..n-1


def test_info_reports_a_pinned_backend_without_contacting_ibm(tmp_path):
    fake = FakeIbm()
    _, prov = provider_for(tmp_path, fake, ibm_backend="ibm_torino")
    assert prov.info()["implementation"] == "ibm_torino"
    assert fake.service_calls == [] and fake.backend_calls == [] and fake.least_busy_calls == []
    assert prov.resolve() == fake.backend.name  # the fake returns its own name; a real pinned backend echoes the pin


def test_info_implementation_is_null_when_unpinned_and_unresolved_then_the_device(tmp_path):
    fake = FakeIbm()
    _, prov = provider_for(tmp_path, fake)
    assert prov.info()["implementation"] is None
    assert fake.service_calls == []
    prov.resolve()
    assert prov.info()["implementation"] == fake.backend.name
