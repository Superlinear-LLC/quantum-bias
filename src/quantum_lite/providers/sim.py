"""Local simulator provider (qiskit-aer)."""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorontoV2

from quantum_lite.circuit import bitstrings_to_matrix, build_circuit, lowest_error_qubits, validate_qubits
from quantum_lite.config import Settings
from quantum_lite.providers.base import Measurement, StateCallback

# 27-qubit fake device: covers the n <= 16 limit (FakeMelbourneV2 has only 15).
FAKE_BACKEND_CLS = FakeTorontoV2


class SimProvider:
    id = "sim"
    variant = "simulator"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._fake = FAKE_BACKEND_CLS() if settings.sim_noise == "fake" else None
        self._implementation = "aer_ideal" if self._fake is None else "aer_fake_toronto"

    def info(self) -> dict:
        return {
            "id": self.id,
            "variant": self.variant,
            "implementation": self._implementation,
            "configured": True,
            "available": True,
            "reason": None,
        }

    def resolve(self) -> str:
        return self._implementation

    def physical_qubits(self, n: int) -> list[int]:
        validate_qubits(n)
        if self._fake is None:
            # Ideal simulator has no measure errors: every tie resolves to the lowest index.
            return list(range(n))
        return lowest_error_qubits(self._fake.target, n)

    def execute(self, circuit: QuantumCircuit, n: int, shots: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """Run `circuit` and return the (shots, n) matrix with the physical qubits used."""
        physical = tuple(self.physical_qubits(n))
        if self._fake is None:
            backend = AerSimulator()
            isa = transpile(circuit, backend)
        else:
            backend = AerSimulator.from_backend(self._fake)
            pm = generate_preset_pass_manager(
                optimization_level=1, backend=self._fake, initial_layout=list(physical),
                translation_method="translator",
            )
            isa = pm.run(circuit)
        result = backend.run(isa, shots=shots, memory=True).result()
        return bitstrings_to_matrix(result.get_memory(), n), physical

    def run(self, n: int, shots: int, on_state: StateCallback, stop: threading.Event) -> Measurement:
        job_id = f"sim-{uuid.uuid4().hex[:12]}"
        backend_name = self._implementation
        on_state("queued", queue_position=None, job_id=job_id, backend=backend_name)
        on_state("running", queue_position=None, job_id=job_id, backend=backend_name)
        latency = self._settings.sim_latency_seconds
        if latency > 0:
            time.sleep(latency)
        bits, physical = self.execute(build_circuit(n), n, shots)
        return Measurement(
            bits=bits,
            job_id=job_id,
            provider=self.id,
            backend=backend_name,
            physical_qubits=physical,
            measured_at=datetime.now(timezone.utc).isoformat(),
            simulated_latency=latency > 0,
        )
