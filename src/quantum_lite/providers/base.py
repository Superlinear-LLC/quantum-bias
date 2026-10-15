"""Provider contract. `run` is blocking and is always called off the event loop."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class Measurement:
    bits: np.ndarray  # (shots, n) uint8, column j = logical qubit j
    job_id: str
    provider: str
    backend: str
    physical_qubits: tuple[int, ...]
    measured_at: str  # ISO-8601 UTC, when the result was received
    simulated_latency: bool = False


# on_state(state, queue_position=..., job_id=..., backend=...)
StateCallback = Callable[..., None]

JOB_STATES = ("submitting", "queued", "initializing", "running", "done", "error", "cancelled")


class Provider(Protocol):
    id: str
    variant: str

    def info(self) -> dict: ...

    def resolve(self) -> str:
        """Implementation name. May contact the network (ibm); call off the event loop."""

    def physical_qubits(self, n: int) -> list[int]: ...

    def run(self, n: int, shots: int, on_state: StateCallback, stop: threading.Event) -> Measurement: ...
