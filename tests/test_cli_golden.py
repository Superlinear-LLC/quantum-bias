"""`quantum-lite prefetch`: stdout, stderr and exit code, line by line (ids by regex).

Pinned before pool feeding moved into pool_feed.py; they pass unchanged after it.
"""
import json
import re

import pytest

from fakes import FakeIbm
from quantum_lite import app as app_mod
from quantum_lite import cli
from quantum_lite.providers.ibm import IbmProvider
from quantum_lite.providers.sim import SimProvider


class CountingIbm(IbmProvider):
    """IbmProvider over the FakeIbm boundary that counts resolve calls."""

    resolves = 0

    def resolve(self) -> str:
        type(self).resolves += 1
        return super().resolve()


@pytest.fixture
def fake(monkeypatch):
    fake = FakeIbm()
    CountingIbm.resolves = 0

    def providers(settings):
        return {"sim": SimProvider(settings),
                "ibm": CountingIbm(settings, service_factory=fake.service_factory,
                                   sampler_cls=fake.sampler_cls, poll_seconds=0.01)}

    monkeypatch.setattr(app_mod, "build_providers", providers)
    return fake


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QL_DATA_DIR", str(tmp_path / "d"))
    return tmp_path / "d"


def run(capsys, *argv):
    code = cli.main(["prefetch", *argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_sim_prefetch_stdout_line_by_line(data_dir, capsys):
    code, out, err = run(capsys, "--provider", "sim", "--qubits", "3", "--shots", "200")
    assert (code, err) == (0, "")
    assert re.fullmatch(
        r"device: aer_ideal\n"
        r"job (sim-[0-9a-f]{12}): queued on aer_ideal\n"
        r"job \1: running on aer_ideal\n"
        r"job_id: \1\n"
        r"shots: 200\n"
        r"estimated QPU seconds: 0 \(simulator\)\n"
        r"pool: sim__aer_ideal__3 now holds 200 unconsumed shots\n", out), out
    code, out, err = run(capsys, "--qubits", "3", "--shots", "50")  # defaults to sim; the pool grows
    assert (code, err) == (0, "")
    assert out.splitlines()[-1] == "pool: sim__aer_ideal__3 now holds 250 unconsumed shots"


def test_ibm_prefetch_without_ibm_backend_prints_the_note(data_dir, fake, monkeypatch, capsys):
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "tok")
    code, out, err = run(capsys, "--provider", "ibm", "--qubits", "3", "--shots", "50")
    assert (code, err) == (0, "")
    assert out == (
        "device: fake_toronto\n"
        "note: IBM_BACKEND is unset; the pool is for fake_toronto. Set IBM_BACKEND=fake_toronto "
        "when you start the app, or it may pick another device and find this pool empty.\n"
        "job fakejob-1: queued on fake_toronto\n"
        "job fakejob-1: running on fake_toronto\n"
        "job_id: fakejob-1\n"
        "shots: 50\n"
        "estimated QPU seconds: 2.0 (estimate: 2 + 0.00035 * shots)\n"
        "pool: ibm__fake_toronto__3 now holds 50 unconsumed shots\n"
    )
    assert CountingIbm.resolves == 1 and fake.jobs == 1


def test_ibm_prefetch_with_ibm_backend_prints_no_note(data_dir, fake, monkeypatch, capsys):
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "tok")
    monkeypatch.setenv("IBM_BACKEND", "fake_toronto")
    code, out, err = run(capsys, "--provider", "ibm", "--qubits", "3", "--shots", "50")
    assert (code, err) == (0, "")
    assert out == (
        "device: fake_toronto\n"
        "job fakejob-1: queued on fake_toronto\n"
        "job fakejob-1: running on fake_toronto\n"
        "job_id: fakejob-1\n"
        "shots: 50\n"
        "estimated QPU seconds: 2.0 (estimate: 2 + 0.00035 * shots)\n"
        "pool: ibm__fake_toronto__3 now holds 50 unconsumed shots\n"
    )


def test_ibm_shot_ceiling_is_one_stderr_line_before_any_ibm_contact(data_dir, fake, monkeypatch, capsys):
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "tok")
    monkeypatch.setenv("IBM_MAX_TOTAL_SHOTS", "10")
    code, out, err = run(capsys, "--provider", "ibm", "--qubits", "3", "--shots", "50")
    assert (code, out) == (1, "")
    assert err == "error: shot_ceiling: IBM_MAX_TOTAL_SHOTS is 10: 0 used + 50 requested\n"
    assert CountingIbm.resolves == 0 and fake.service_calls == [] and fake.jobs == 0
    assert not (data_dir / "pools").exists()  # the pool store is opened after device:, never reached


@pytest.mark.parametrize("argv, env, err", [
    (["--provider", "nope", "--qubits", "17"], {},
     "error: unknown_provider: unknown provider 'nope'; known: ['ibm', 'sim']\n"),
    (["--provider", "ibm", "--qubits", "17"], {},
     "error: provider_unavailable: provider ibm unavailable: IBM_QUANTUM_TOKEN not set\n"),
    (["--qubits", "17", "--shots", "0"], {},
     "error: invalid_num_qubits: num_qubits must be 1..16, got 17\n"),
    (["--qubits", "0"], {}, "error: invalid_num_qubits: num_qubits must be 1..16, got 0\n"),
    (["--shots", "0"], {}, "error: invalid_shots: shots must be >= 1, got 0\n"),
    (["--shots", "1000001"], {}, "error: invalid_shots: shots must be 1..1000000, got 1000001\n"),
    (["--provider", "ibm", "--shots", "0"], {"IBM_QUANTUM_TOKEN": "tok"},
     "error: invalid_shots: shots must be >= 1, got 0\n"),
    (["--provider", "ibm", "--shots", "20001"], {"IBM_QUANTUM_TOKEN": "tok"},
     "error: shots_exceed_job_ceiling: 20001 shots requested, IBM_MAX_SHOTS_PER_JOB is 20000\n"),
])
def test_refused_prefetch_is_one_stderr_line_and_exit_1(data_dir, fake, monkeypatch, capsys, argv, env, err):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    code, out, got = run(capsys, *argv)
    assert (code, out, got) == (1, "", err)
    assert CountingIbm.resolves == 0 and fake.service_calls == [] and fake.jobs == 0


def test_corrupt_pool_file_fails_after_device_is_printed(data_dir, capsys):
    (data_dir / "pools").mkdir(parents=True)
    (data_dir / "pools" / "sim__aer_ideal__3.json").write_text(json.dumps({"provider": "sim"}))
    with pytest.raises(KeyError):
        cli.main(["prefetch", "--qubits", "3", "--shots", "10"])
    out, err = capsys.readouterr()
    assert (out, err) == ("device: aer_ideal\n", "")
