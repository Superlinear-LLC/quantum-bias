"""The circuit IbmProvider submits passes qiskit-ibm-runtime's own ISA check (no network, no token).

tests/fakes.py bypasses the runtime's pre-submit validation, so this runs the real
validate_isa_circuits on what the provider actually builds and transpiles for a Heron-class target.
"""
import threading

import pytest
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime.utils import validate_isa_circuits

from conftest import make_settings
from quantum_lite.providers.ibm import IbmProvider


class _Submitted(Exception):
    pass


def test_provider_circuit_passes_real_isa_validation(tmp_path):
    backend = FakeFez()
    captured = []

    class CapturingSampler:
        def __init__(self, mode):
            assert mode is backend

        def run(self, pubs, shots):
            captured.extend(pubs)
            raise _Submitted

    class Service:
        def __init__(self, **kw):
            pass

        def least_busy(self, **kw):
            return backend

    prov = IbmProvider(make_settings(tmp_path, ibm_token="T"), service_factory=Service,
                       sampler_cls=CapturingSampler, poll_seconds=0.01)
    with pytest.raises(_Submitted):
        prov.run(3, 10, lambda *a, **k: None, threading.Event())

    assert len(captured) == 1
    # Raises IBMInputValueError for a non-ISA circuit; returns None when accepted.
    assert validate_isa_circuits([captured[0]], backend.target) is None
