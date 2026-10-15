import threading

import numpy as np
import pytest

from quantum_lite.circuit import bitstrings_to_matrix, build_circuit
from quantum_lite.config import Settings
from quantum_lite.providers.sim import SimProvider

SHOTS = 4000
# 50% column of N shots: sigma = 0.5/sqrt(N) = 0.0079; 0.05 is > 6 sigma.
TOL = 0.05


def test_bitstring_reversal_is_deterministic():
    # Qiskit writes qubit 0 rightmost: '001' means qubit 0 = 1.
    assert bitstrings_to_matrix(["001", "100"], 3).tolist() == [[1, 0, 0], [0, 0, 1]]


@pytest.mark.parametrize("noise", ["none", "fake"])
def test_x_on_qubit_zero_gives_column_zero_ones(tmp_path, noise):
    sim = SimProvider(Settings(data_dir=tmp_path, sim_noise=noise))
    bits, physical = sim.execute(build_circuit(3, x_on=[0]), 3, SHOTS)
    assert bits.shape == (SHOTS, 3)
    col0 = bits[:, 0].mean()
    if noise == "none":
        assert col0 == 1.0
    else:
        assert col0 > 0.8  # readout error on the fake device
    assert abs(bits[:, 1].mean() - 0.5) < TOL
    assert abs(bits[:, 2].mean() - 0.5) < TOL


def test_x_on_last_qubit_is_not_confused_with_first(tmp_path):
    sim = SimProvider(Settings(data_dir=tmp_path))
    bits, _ = sim.execute(build_circuit(3, x_on=[2]), 3, SHOTS)
    assert bits[:, 2].mean() == 1.0
    assert abs(bits[:, 0].mean() - 0.5) < TOL
