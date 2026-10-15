"""Real QPU smoke test. Never runs in CI: needs QL_LIVE_IBM=1 and a token in the environment."""
import dataclasses
import os

import pytest
from fastapi.testclient import TestClient

from conftest import REAL_ENV_FILE, new_session
from quantum_lite.app import create_app
from quantum_lite.config import load_settings

def _real_env():
    return REAL_ENV_FILE if REAL_ENV_FILE.is_file() else None


# The token and IBM_BACKEND come from the same place the app reads them: process env over .env.
# The smoke test must behave like the running app: third-party deprecations on IBM's submit/poll path are
# reported in the warnings summary, not raised. The unit suite keeps warnings as errors.
pytestmark = [
    pytest.mark.filterwarnings("default"),
    pytest.mark.skipif(
        os.environ.get("QL_LIVE_IBM") != "1" or not load_settings(_real_env()).ibm_configured,
        reason="set QL_LIVE_IBM=1 and a token (env or .env) to run against a real QPU (bills QPU time)",
    ),
]


def test_one_live_job(tmp_path):
    settings = dataclasses.replace(load_settings(_real_env()), data_dir=tmp_path / "d", ibm_max_jobs=1, ibm_max_total_shots=100)
    with TestClient(create_app(settings)) as c:
        s = new_session(c, provider="ibm", mode="live", num_qubits=2, shots_per_job=10)
        r = c.post(f"/api/sessions/{s['id']}/press")
        assert r.status_code == 200, r.text
        assert r.json()["provenance"]["provider"] == "ibm"
