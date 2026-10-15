"""The independent_qubits circuit and bit-order handling."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from quantum_lite.errors import AppError, ProviderError

MIN_QUBITS = 1
MAX_QUBITS = 16
CREG_NAME = "c"


def validate_qubits(n: int) -> None:
    if not (MIN_QUBITS <= n <= MAX_QUBITS):
        raise AppError(400, "invalid_num_qubits", f"num_qubits must be {MIN_QUBITS}..{MAX_QUBITS}, got {n}")


def build_circuit(n: int, x_on: Iterable[int] = ()) -> QuantumCircuit:
    """H on each of n qubits, measure into classical register `c` of size n.

    `x_on` (bit-order test only) puts X instead of H on those qubits, so they read 1 with
    certainty while the others stay at 50%.
    """
    validate_qubits(n)
    flipped = set(x_on)
    qc = QuantumCircuit(QuantumRegister(n, "q"), ClassicalRegister(n, CREG_NAME))
    for q in range(n):
        if q in flipped:
            qc.x(q)
        else:
            qc.h(q)
    qc.measure(range(n), range(n))
    return qc


def bitstrings_to_matrix(bitstrings: Sequence[str], n: int) -> np.ndarray:
    """Qiskit bitstrings are little-endian (qubit 0 rightmost): reverse to column j = qubit j."""
    if len(bitstrings) == 0:
        raise ValueError("no bitstrings to convert")
    arr = np.array([[int(ch) for ch in s.replace(" ", "")] for s in bitstrings], dtype=np.uint8)
    if arr.shape[1] != n:
        raise ValueError(f"bitstrings have {arr.shape[1]} bits, expected {n}")
    return np.ascontiguousarray(arr[:, ::-1])


def lowest_error_qubits(target, n: int) -> list[int]:
    """The n qubits with lowest measure error in target['measure'] (ties -> lowest index).

    Only qubits that can also run the gates the transpiled circuit needs (sx and rz) are
    candidates: a pure filter, never a substitute. Missing data raises ProviderError.
    """
    measure = target["measure"]
    errors: list[tuple[float, int]] = []
    for qargs, props in measure.items():
        if qargs is None or len(qargs) != 1:
            continue
        if props is None or props.error is None:
            raise ProviderError(f"target has no measure error for qubit {qargs}")
        if not all(target.instruction_supported(gate, tuple(qargs)) for gate in ("sx", "rz")):
            continue
        errors.append((float(props.error), int(qargs[0])))
    if len(errors) < n:
        raise ProviderError(f"backend exposes {len(errors)} usable measurable qubits (measure, sx, rz), need {n}")
    chosen = sorted(errors)[:n]
    return sorted(q for _, q in chosen)
