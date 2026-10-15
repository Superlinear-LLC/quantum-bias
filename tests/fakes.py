"""Fakes for the provider boundary.

FakeIbm: QiskitRuntimeService, SamplerV2 and the runtime job. The fake sampler really executes
the transpiled circuit on Aer, so bit strings have genuine qiskit (little-endian) layout and the
IBM code path's conversion is exercised for real.

ScriptedSim: the simulator provider with known bits. Everything around the measurement (the job
record and its states, the runner, provenance, aggregation, the stream loop) is the real code.

A mock proves only the boundary it models: these model the provider, nothing above it.
"""
from __future__ import annotations

import collections
import threading
import time
from types import SimpleNamespace

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorontoV2

from quantum_lite.errors import ProviderError
from quantum_lite.providers.sim import SimProvider


class FakeIbmBackend(FakeTorontoV2):
    """FakeTorontoV2 plus the hook a real IBMBackend has (ibm_backend.py get_translation_stage_plugin).

    Without translation_method the pass manager asks this hook for a plugin that cannot import
    on qiskit 1.4.5, which is what breaks every real IBM job.
    """

    def get_translation_stage_plugin(self):
        return "ibm_dynamic_circuits"


class FakeJob:
    def __init__(self, job_id, statuses, bitstrings, error="boom"):
        self._id = job_id
        self._statuses = list(statuses)
        self._bitstrings = bitstrings
        self._error = error
        self._final = self._statuses[-1]

    def job_id(self):
        return self._id

    def status(self):
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def error_message(self):
        # As RuntimeJobV2 0.44.1 (base_runtime_job._set_error_message): only an ERROR job has one.
        return self._error if self._final == "ERROR" else None

    def result(self):
        data = SimpleNamespace(c=SimpleNamespace(get_bitstrings=lambda: self._bitstrings))
        return [SimpleNamespace(data=data)]


class FakeIbm:
    """Holds recorded calls and scripted behaviour; its methods are the injected factories."""

    def __init__(self, statuses=("QUEUED", "RUNNING", "DONE"), error="boom"):
        self.statuses = statuses
        self.error = error
        self.service_calls: list[dict] = []
        self.least_busy_calls: list[dict] = []
        self.backend_calls: list[str] = []
        self.sampler_runs: list[dict] = []
        self.circuits: list = []  # the ISA circuit of every submitted job
        self.backend = FakeIbmBackend()
        self.jobs = 0

    def service_factory(self, **kwargs):
        self.service_calls.append(kwargs)
        outer = self

        class Service:
            def least_busy(self, **kw):
                outer.least_busy_calls.append(kw)
                return outer.backend

            def backend(self, name):
                outer.backend_calls.append(name)
                return outer.backend

            def save_account(self, *a, **k):
                raise AssertionError("save_account must never be called")

        return Service()

    def sampler_cls(self, mode):
        outer = self

        class Sampler:
            def run(self, pubs, shots):
                outer.jobs += 1
                outer.sampler_runs.append({"mode": mode, "shots": shots})
                circuit = pubs[0]
                outer.circuits.append(circuit)
                result = AerSimulator().run(circuit, shots=shots, memory=True).result()
                return FakeJob(f"fakejob-{outer.jobs}", outer.statuses, result.get_memory(), outer.error)

        return Sampler()


class ScriptedSim(SimProvider):
    """A simulator whose jobs measure the scripted bits, one entry per job, in order.

    An entry is a (shots, n) matrix, or one bit per shot (the same bit on every qubit). An
    Exception entry is raised by that job, in the provider thread: a real job error. Each job
    first sleeps `delays.pop(0)` if any are left, else `delay` seconds (in the provider thread, so
    the event loop stays free). With the script used up a job raises ProviderError("no scripted
    bits left"), or with when_empty="block" waits until shutdown.
    """

    def __init__(self, settings, script=(), *, delay=0.0, delays=(), when_empty="raise"):
        super().__init__(settings)
        assert when_empty in ("raise", "block")
        self.script = collections.deque(script)
        self.delay = delay
        self.delays = collections.deque(delays)
        self.when_empty = when_empty
        self.jobs = 0
        self._stop = threading.Event()

    def run(self, n, shots, on_state, stop):
        self._stop = stop  # the runner's: set at shutdown
        return super().run(n, shots, on_state, stop)

    def execute(self, circuit, n, shots):
        self.jobs += 1
        time.sleep(self.delays.popleft() if self.delays else self.delay)
        if not self.script:
            if self.when_empty == "block":
                self._stop.wait(60)  # bounded: a test that never shuts down must not hang the interpreter exit
                raise ProviderError("shutdown")
            raise ProviderError("no scripted bits left")
        entry = self.script.popleft()
        if isinstance(entry, Exception):
            raise entry
        bits = np.asarray(entry, dtype=np.uint8)
        if bits.ndim == 1:
            bits = np.repeat(bits[:, None], n, axis=1)
        if bits.shape != (shots, n):
            raise ProviderError(f"scripted bits have shape {bits.shape}, the job is {shots} shots of {n} qubits")
        return bits, tuple(self.physical_qubits(n))
