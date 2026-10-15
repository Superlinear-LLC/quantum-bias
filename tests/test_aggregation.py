import numpy as np
import pytest

from quantum_lite.aggregation import aggregate
from quantum_lite.errors import AppError


M = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)


def test_none_takes_qubit_zero():
    assert aggregate(M, "none").tolist() == [1, 1, 0, 0]


def test_xor_is_parity():
    assert aggregate(M, "xor").tolist() == [0, 1, 1, 0]


def test_majority_odd():
    assert aggregate(M, "majority").tolist() == [1, 1, 0, 0]


def test_majority_even_n_rejected():
    with pytest.raises(AppError) as e:
        aggregate(np.zeros((2, 4), dtype=np.uint8), "majority")
    assert e.value.status == 400


def test_unknown_aggregation_rejected():
    with pytest.raises(AppError):
        aggregate(M, "median")
