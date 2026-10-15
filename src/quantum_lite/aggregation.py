"""Aggregation of one shot (n bits) into one bit.

xor (parity) and majority vote, vectorised over a (shots, n) matrix. Majority is only
allowed for odd n, so a tie cannot occur.
"""
from __future__ import annotations

import numpy as np

from quantum_lite.errors import AppError

AGGREGATIONS = ("none", "xor", "majority")


def validate_aggregation(aggregation: str, n_qubits: int) -> None:
    if aggregation not in AGGREGATIONS:
        raise AppError(400, "invalid_aggregation", f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    if aggregation == "majority" and n_qubits % 2 == 0:
        raise AppError(400, "invalid_aggregation", f"majority requires an odd number of qubits, got {n_qubits}")


def aggregate(matrix: np.ndarray, aggregation: str) -> np.ndarray:
    """(shots, n) uint8 -> (shots,) uint8."""
    if matrix.ndim != 2:
        raise ValueError(f"expected a (shots, n) matrix, got shape {matrix.shape}")
    n = matrix.shape[1]
    validate_aggregation(aggregation, n)
    if aggregation == "none":
        return matrix[:, 0].astype(np.uint8)
    if aggregation == "xor":
        return (matrix.sum(axis=1, dtype=np.int64) % 2).astype(np.uint8)
    return (matrix.sum(axis=1, dtype=np.int64) * 2 > n).astype(np.uint8)
