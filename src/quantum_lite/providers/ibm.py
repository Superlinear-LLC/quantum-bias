"""IBM Quantum provider (qiskit-ibm-runtime 0.44.1, qiskit 1.4.5).

Confirmed against the installed sources:
- QiskitRuntimeService(channel, token, ..., instance) in qiskit_runtime_service.py.
- least_busy(min_num_qubits=None, instance=None, filters=None, **kwargs); `operational` and
  `simulator` pass through **kwargs to backends().
- SamplerV2(mode=backend) in job mode; sampler.run([pub], shots=int) -> RuntimeJobV2.
- RuntimeJobV2.status() returns a qiskit JobStatus enum (map_status reads .name), with QUEUED,
  RUNNING, DONE, ERROR, CANCELLED from the API; INITIALIZING is only the constructor's initial value.
- RuntimeJobV2 has no queue_position() in 0.44.1; queue position is reported as null.
- result: PrimitiveResult[0].data.<creg>.get_bitstrings(); our creg is `c`.
- backend.target["measure"][(q,)].error is the per-qubit readout error.
- generate_preset_pass_manager(optimization_level, backend, ..., initial_layout=...).

The token goes straight to the QiskitRuntimeService constructor; save_account is never called
and nothing is ever persisted.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from quantum_lite.circuit import CREG_NAME, bitstrings_to_matrix, build_circuit, lowest_error_qubits
from quantum_lite.config import Settings
from quantum_lite.errors import ProviderError
from quantum_lite.providers.base import Measurement, StateCallback

log = logging.getLogger("quantum_lite.ibm")

NO_TOKEN_REASON = "IBM_QUANTUM_TOKEN not set"
POLL_SECONDS = 2.0
_STATE_MAP = {
    "INITIALIZING": "initializing",
    "QUEUED": "queued",
    "RUNNING": "running",
    "DONE": "done",
    "ERROR": "error",
    "CANCELLED": "cancelled",
}


def map_status(status: Any) -> str:
    """Map a runtime job status to ours. Unknown values raise; they are never guessed."""
    key = getattr(status, "name", status)
    if not isinstance(key, str) or key.upper() not in _STATE_MAP:
        raise ProviderError(f"unrecognised IBM job status: {status!r}")
    return _STATE_MAP[key.upper()]


class IbmProvider:
    id = "ibm"
    variant = "qpu"

    def __init__(
        self,
        settings: Settings,
        *,
        service_factory: Callable[..., Any] = QiskitRuntimeService,
        sampler_cls: Callable[..., Any] = SamplerV2,
        pass_manager_factory: Callable[..., Any] = generate_preset_pass_manager,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._settings = settings
        self._service_factory = service_factory
        self._sampler_cls = sampler_cls
        self._pm_factory = pass_manager_factory
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._service: Any = None
        self._backend: Any = None
        self._layouts: dict[tuple[str, int], list[int]] = {}

    # --- discovery -------------------------------------------------------------------
    def info(self) -> dict:
        configured = self._settings.ibm_configured
        # A pinned IBM_BACKEND is the name _backend_locked resolves, so it is reported without
        # contacting IBM. Unpinned, it stays null until the first use picks a device.
        if self._backend is not None:
            implementation = self._backend.name
        else:
            implementation = self._settings.ibm_backend or None
        return {
            "id": self.id,
            "variant": self.variant,
            "implementation": implementation,
            "configured": configured,
            "available": configured,
            "reason": None if configured else NO_TOKEN_REASON,
        }

    def _require_token(self) -> str:
        token = self._settings.ibm_token
        if token is None:
            raise ProviderError(NO_TOKEN_REASON)
        return token

    def _backend_locked(self) -> Any:
        """Create the service and resolve the backend once per process. Caller holds the lock."""
        if self._backend is not None:
            return self._backend
        s = self._settings
        service = self._service_factory(channel=s.ibm_channel, token=self._require_token(), instance=s.ibm_instance)
        if s.ibm_backend:
            backend = service.backend(s.ibm_backend)
        else:
            backend = service.least_busy(operational=True, simulator=False)
        self._service = service
        self._backend = backend
        log.info("resolved IBM backend %s", backend.name)
        return backend

    def resolve(self) -> str:
        with self._lock:
            return self._backend_locked().name

    def physical_qubits(self, n: int) -> list[int]:
        with self._lock:
            backend = self._backend_locked()
            key = (backend.name, n)
            if key not in self._layouts:
                self._layouts[key] = lowest_error_qubits(backend.target, n)
            return list(self._layouts[key])

    # --- execution -------------------------------------------------------------------
    def run(self, n: int, shots: int, on_state: StateCallback, stop: threading.Event) -> Measurement:
        physical = self.physical_qubits(n)
        with self._lock:
            backend = self._backend_locked()
        # translation_method is pinned: a real IBMBackend names the ibm_dynamic_circuits plugin
        # through get_translation_stage_plugin(), and that plugin cannot import on qiskit 1.4.5.
        pm = self._pm_factory(optimization_level=1, backend=backend, initial_layout=physical,
                              translation_method="translator")
        isa = pm.run(build_circuit(n))
        sampler = self._sampler_cls(mode=backend)
        job = sampler.run([isa], shots=shots)
        job_id = job.job_id()
        log.info("submitted IBM job %s (%d shots, %d qubits) on %s", job_id, shots, n, backend.name)

        last = None  # the first state published is the first one IBM reports, polled right now
        while True:
            try:
                state = map_status(job.status())
            except ProviderError as exc:
                # The job is already on IBM: name it so it can be fetched by hand.
                raise ProviderError(f"IBM job {job_id}: {exc}") from exc
            queue_position = job.queue_position() if hasattr(job, "queue_position") else None
            if state != last and state != "done":  # the runner publishes done after the result
                on_state(state, queue_position=queue_position, job_id=job_id, backend=backend.name)
                last = state
            if state == "done":
                break
            if state in ("error", "cancelled"):
                raise ProviderError(f"IBM job {job_id} ended {state}: {_job_error(job)}")
            if stop.is_set():
                raise ProviderError(f"shutdown while IBM job {job_id} was {state}; the job stays on IBM")
            time.sleep(self._poll_seconds)

        bitstrings = getattr(job.result()[0].data, CREG_NAME).get_bitstrings()
        if len(bitstrings) != shots:
            raise ProviderError(f"IBM job {job_id} returned {len(bitstrings)} shots, expected {shots}")
        return Measurement(
            bits=bitstrings_to_matrix(bitstrings, n),
            job_id=job_id,
            provider=self.id,
            backend=backend.name,
            physical_qubits=tuple(physical),
            measured_at=datetime.now(timezone.utc).isoformat(),
        )


def _job_error(job: Any) -> str:
    message = job.error_message()
    return "no error message reported" if message is None else str(message)
