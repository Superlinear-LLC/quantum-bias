"""Bias statistics.

Bias percent and sigma (z) of a bit stream:
  bias_pct = (k - N/2) / (N/2) * 100,  z = (k - N/2) / (sqrt(N)/2).
With N = 0 both are 0.0.
"""
from __future__ import annotations

import math

import numpy as np

WINDOW = 1000


def bias_pct(ones: int, n: int) -> float:
    if n == 0:
        return 0.0
    return (ones - n / 2) / (n / 2) * 100.0


def z_score(ones: int, n: int) -> float:
    if n == 0:
        return 0.0
    return (ones - n / 2) / (math.sqrt(n) / 2)


class BitCounter:
    """Counts a stream of bits: totals plus a sliding window of the last WINDOW bits."""

    def __init__(self) -> None:
        self.n = 0
        self.ones = 0
        self._window = np.zeros(0, dtype=np.uint8)

    def add(self, bits: np.ndarray) -> None:
        flat = np.asarray(bits, dtype=np.uint8).ravel()
        self.n += int(flat.size)
        self.ones += int(flat.sum(dtype=np.int64))
        self._window = np.concatenate([self._window, flat])[-WINDOW:]

    def snapshot(self) -> dict:
        wn = int(self._window.size)
        wk = int(self._window.sum(dtype=np.int64))
        return {
            "n": self.n,
            "ones": self.ones,
            "bias_pct": bias_pct(self.ones, self.n),
            "z": z_score(self.ones, self.n),
            "window": {"n": wn, "ones": wk, "bias_pct": bias_pct(wk, wn), "z": z_score(wk, wn)},
        }


class HitCounter:
    """Predict / Perceive trials."""

    def __init__(self) -> None:
        self.trials = 0
        self.hits = 0
        self.streak = 0
        self.best_streak = 0

    def add(self, hit: bool) -> None:
        self.trials += 1
        if hit:
            self.hits += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = 0

    def snapshot(self) -> dict:
        return {
            "trials": self.trials,
            "hits": self.hits,
            # None, not 0.0: there is no rate without trials.
            "hit_rate": None if self.trials == 0 else self.hits / self.trials,
            "z_hits": z_score(self.hits, self.trials),
            "streak": self.streak,
            "best_streak": self.best_streak,
        }


class Stats:
    """Stats for one scope (a whole session, or one mode segment)."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.shots = 0
        self.raw = BitCounter()
        self.aggregated = BitCounter()
        self.qubit_ones = np.zeros(n_qubits, dtype=np.int64)
        self.predict = HitCounter()
        self.perceive = HitCounter()

    def add_block(self, matrix: np.ndarray, aggregated: np.ndarray) -> None:
        if matrix.ndim != 2 or matrix.shape[1] != self.n_qubits:
            raise ValueError(f"expected (shots, {self.n_qubits}) bits, got {matrix.shape}")
        if aggregated.shape != (matrix.shape[0],):
            raise ValueError(f"aggregated shape {aggregated.shape} does not match {matrix.shape}")
        self.shots += int(matrix.shape[0])
        self.raw.add(matrix)
        self.aggregated.add(aggregated)
        self.qubit_ones += matrix.sum(axis=0, dtype=np.int64)

    def snapshot(self) -> dict:
        return {
            "shots": self.shots,
            "raw": self.raw.snapshot(),
            "aggregated": self.aggregated.snapshot(),
            # None per qubit until a shot exists.
            "per_qubit_ones_fraction": [
                None if self.shots == 0 else float(k) / self.shots for k in self.qubit_ones
            ],
            "predict": self.predict.snapshot(),
            "perceive": self.perceive.snapshot(),
        }
