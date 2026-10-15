import sys
from pathlib import Path

import pytest

from quantum_lite import cli, config
from quantum_lite.config import Settings, load_settings
from quantum_lite.errors import AppError, ConfigError


def test_defaults():
    s = Settings.from_env({})
    assert s.ibm_token is None and s.ibm_configured is False
    assert (s.ibm_channel, s.ibm_max_shots_per_job, s.ibm_max_total_shots, s.ibm_max_jobs) == \
        ("ibm_quantum_platform", 20000, 200000, 100)
    assert s.sim_noise == "none" and s.sim_latency_seconds == 0.0
    assert s.data_dir.name == ".data" and s.data_dir.parent == config.PROJECT_ROOT


def test_token_alias_and_empty_means_unset():
    assert Settings.from_env({"QISKIT_IBM_TOKEN": "a"}).ibm_token == "a"
    assert Settings.from_env({"IBM_QUANTUM_TOKEN": "b", "QISKIT_IBM_TOKEN": "a"}).ibm_token == "b"
    assert Settings.from_env({"IBM_QUANTUM_TOKEN": ""}).ibm_configured is False
    assert "SECRET" not in repr(Settings.from_env({"IBM_QUANTUM_TOKEN": "SECRET"}))


@pytest.mark.parametrize("env", [
    {"IBM_MAX_JOBS": "many"}, {"IBM_MAX_JOBS": "0"}, {"SIM_NOISE": "loud"},
    {"SIM_LATENCY_SECONDS": "-1"}, {"SIM_LATENCY_SECONDS": "x"},
])
def test_bad_values_raise(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_env_file_and_precedence(tmp_path, monkeypatch):
    f = tmp_path / "e.env"
    f.write_text("IBM_MAX_JOBS=7\nSIM_NOISE=fake\n")
    monkeypatch.delenv("IBM_MAX_JOBS", raising=False)
    monkeypatch.setenv("SIM_NOISE", "none")
    s = load_settings(f)
    assert s.ibm_max_jobs == 7 and s.sim_noise == "none"  # process env wins
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "missing.env")


def test_cli_pools_and_prefetch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QL_DATA_DIR", str(tmp_path / "d"))
    assert cli.main(["pools"]) == 0
    assert "no pools" in capsys.readouterr().out
    assert cli.main(["prefetch", "--provider", "sim", "--qubits", "3", "--shots", "200"]) == 0
    out = capsys.readouterr().out
    assert "job_id: sim-" in out and "shots: 200" in out and "estimated QPU seconds" in out
    assert cli.main(["pools"]) == 0
    assert "sim__aer_ideal__3\tavailable=200" in capsys.readouterr().out


def test_cli_prefetch_ibm_without_token_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QL_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    assert cli.main(["prefetch", "--provider", "ibm"]) == 1
    err = capsys.readouterr().err.strip()
    assert err == "error: provider_unavailable: provider ibm unavailable: IBM_QUANTUM_TOKEN not set"  # one line


def test_cli_errors_are_one_line_but_unexpected_errors_propagate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QL_DATA_DIR", str(tmp_path / "d"))
    assert cli.main(["prefetch", "--qubits", "17"]) == 1
    assert capsys.readouterr().err.startswith("error: invalid_num_qubits:")
    assert cli.main(["--env-file", str(tmp_path / "nope.env"), "pools"]) == 1
    assert capsys.readouterr().err.startswith("error: config_error:")
    assert cli.main(["prefetch", "--shots", "1000001"]) == 1
    assert "invalid_shots" in capsys.readouterr().err
    monkeypatch.setattr(cli, "_pools", lambda settings: (_ for _ in ()).throw(RuntimeError("bug")))
    with pytest.raises(RuntimeError, match="bug"):
        cli.main(["pools"])


def test_cli_port_with_help_is_serve_help(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--port", "8766", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--port" in out and "--open" in out
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "prefetch" in capsys.readouterr().out


def test_cli_prefetch_prints_device_and_ibm_backend_hint(tmp_path, monkeypatch, capsys):
    from fakes import FakeIbm
    from quantum_lite import app as app_mod
    from quantum_lite.providers.ibm import IbmProvider
    from quantum_lite.providers.sim import SimProvider

    monkeypatch.setenv("QL_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "tok")
    monkeypatch.delenv("IBM_BACKEND", raising=False)
    fake = FakeIbm()

    def providers(settings):
        return {"sim": SimProvider(settings),
                "ibm": IbmProvider(settings, service_factory=fake.service_factory,
                                   sampler_cls=fake.sampler_cls, poll_seconds=0.01)}

    monkeypatch.setattr(app_mod, "build_providers", providers)
    assert cli.main(["prefetch", "--provider", "ibm", "--qubits", "3", "--shots", "50"]) == 0
    out = capsys.readouterr().out
    assert f"device: {fake.backend.name}" in out and f"IBM_BACKEND={fake.backend.name}" in out
    monkeypatch.setenv("IBM_BACKEND", fake.backend.name)
    assert cli.main(["prefetch", "--provider", "ibm", "--qubits", "3", "--shots", "50"]) == 0
    out = capsys.readouterr().out
    assert f"device: {fake.backend.name}" in out and "note:" not in out


def test_cli_serve_defaults_flags_and_non_local_warning(tmp_path, monkeypatch, capsys):
    import uvicorn

    monkeypatch.setenv("QL_DATA_DIR", str(tmp_path / "d"))
    calls = []
    monkeypatch.setattr(uvicorn.Server, "run",
                        lambda self, *a, **k: calls.append({"host": self.config.host, "port": self.config.port}))
    assert cli.main([]) == 0
    assert calls[-1]["host"] == "127.0.0.1" and calls[-1]["port"] == 8765
    assert "WARNING" not in capsys.readouterr().err
    assert cli.main(["--port", "9000", "--host", "0.0.0.0"]) == 0
    assert calls[-1]["port"] == 9000
    assert "WARNING" in capsys.readouterr().err
    assert cli.main(["serve", "--port", "9001"]) == 0 and calls[-1]["port"] == 9001


def test_ibm_channel_local_is_rejected():
    # "local" makes QiskitRuntimeService a local fake-backend simulator: its bits would be labelled ibm.
    with pytest.raises(ConfigError, match="IBM_QUANTUM_CHANNEL"):
        Settings.from_env({"IBM_QUANTUM_TOKEN": "x", "IBM_QUANTUM_CHANNEL": "local"})
    assert Settings.from_env({"IBM_QUANTUM_CHANNEL": "ibm_cloud"}).ibm_channel == "ibm_cloud"
    assert Settings.from_env({}).ibm_channel == "ibm_quantum_platform"


def test_pytest_ignores_the_users_environment(tmp_path):
    """IBM_BACKEND / SIM_NOISE in the environment (or a token in .env) must not
    reach the suite. Run the CLI tests in a child pytest whose environment is polluted."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "IBM_BACKEND": "ibm_leak", "SIM_NOISE": "fake", "IBM_MAX_JOBS": "1"}
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
                          str(root / "tests" / "test_config_cli.py"),
                          "-k", "prefetch_prints_device or pools_and_prefetch or without_token"],
                         env=env, cwd=root, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_default_env_file_is_not_the_users():
    from conftest import REAL_ENV_FILE
    from quantum_lite import config

    assert config.DEFAULT_ENV_FILE != REAL_ENV_FILE and not config.DEFAULT_ENV_FILE.exists()
