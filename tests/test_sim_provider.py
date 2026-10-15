import threading

import pytest

from quantum_lite.circuit import lowest_error_qubits
from quantum_lite.config import Settings
from quantum_lite.errors import AppError
from quantum_lite.providers.sim import FAKE_BACKEND_CLS, SimProvider


def run(sim, n, shots):
    states = []
    m = sim.run(n, shots, lambda state, **kw: states.append((state, kw)), threading.Event())
    return m, states


def test_ideal_shapes_and_provenance(tmp_path):
    sim = SimProvider(Settings(data_dir=tmp_path))
    m, states = run(sim, 4, 300)
    assert m.bits.shape == (300, 4) and m.bits.dtype.name == "uint8"
    assert set(m.bits.ravel().tolist()) <= {0, 1}
    assert m.physical_qubits == (0, 1, 2, 3)
    assert m.provider == "sim" and m.backend == "aer_ideal"
    assert m.job_id.startswith("sim-") and m.measured_at
    assert m.simulated_latency is False
    assert [s for s, _ in states] == ["queued", "running"]
    assert states[0][1]["job_id"] == m.job_id and states[0][1]["queue_position"] is None


def test_latency_is_reported_and_applied(tmp_path):
    sim = SimProvider(Settings(data_dir=tmp_path, sim_latency_seconds=0.3))
    import time
    t0 = time.monotonic()
    m, _ = run(sim, 2, 10)
    assert time.monotonic() - t0 >= 0.3
    assert m.simulated_latency is True


def test_noisy_sim_uses_lowest_measure_error_qubits(tmp_path):
    sim = SimProvider(Settings(data_dir=tmp_path, sim_noise="fake"))
    target = FAKE_BACKEND_CLS().target
    errors = {q[0]: p.error for q, p in target["measure"].items()}
    expected = sorted(sorted(errors, key=lambda q: (errors[q], q))[:5])
    m, _ = run(sim, 5, 200)
    assert list(m.physical_qubits) == expected == lowest_error_qubits(target, 5)
    assert m.backend == "aer_fake_toronto"
    assert m.bits.shape == (200, 5)


def test_qubit_limits(tmp_path):
    sim = SimProvider(Settings(data_dir=tmp_path))
    with pytest.raises(AppError):
        sim.physical_qubits(17)
    with pytest.raises(AppError):
        sim.physical_qubits(0)
    m, _ = run(sim, 16, 20)
    assert m.bits.shape == (20, 16)
