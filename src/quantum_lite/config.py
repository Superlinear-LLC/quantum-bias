"""Environment configuration. Pure: parses a mapping, never mutates os.environ."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from quantum_lite.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"
SIM_NOISE_VALUES = ("none", "fake")
# qiskit-ibm-runtime also accepts channel "local", which returns a local fake-backend simulator
# whose results would be labelled IBM QPU data. Only the two real services are allowed.
IBM_CHANNELS = ("ibm_quantum_platform", "ibm_cloud")


@dataclass(frozen=True)
class Settings:
    # repr=False: the token must never reach a log line, traceback or repr.
    ibm_token: str | None = field(default=None, repr=False)
    ibm_instance: str | None = None
    ibm_channel: str = "ibm_quantum_platform"
    ibm_backend: str | None = None
    ibm_max_shots_per_job: int = 20000
    ibm_max_total_shots: int = 200000
    ibm_max_jobs: int = 100
    sim_noise: str = "none"
    sim_latency_seconds: float = 0.0
    data_dir: Path = PROJECT_ROOT / ".data"

    @property
    def ibm_configured(self) -> bool:
        return self.ibm_token is not None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        def get(name: str) -> str | None:
            value = env.get(name)
            if value is None or value.strip() == "":
                return None
            return value.strip()

        def get_int(name: str, default: int) -> int:
            raw = get(name)
            if raw is None:
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
            if value < 1:
                raise ConfigError(f"{name} must be >= 1, got {value}")
            return value

        latency_raw = get("SIM_LATENCY_SECONDS")
        try:
            latency = 0.0 if latency_raw is None else float(latency_raw)
        except ValueError as exc:
            raise ConfigError(f"SIM_LATENCY_SECONDS must be a number, got {latency_raw!r}") from exc
        if latency < 0:
            raise ConfigError(f"SIM_LATENCY_SECONDS must be >= 0, got {latency}")

        noise = get("SIM_NOISE") or "none"
        if noise not in SIM_NOISE_VALUES:
            raise ConfigError(f"SIM_NOISE must be one of {SIM_NOISE_VALUES}, got {noise!r}")

        channel = get("IBM_QUANTUM_CHANNEL") or "ibm_quantum_platform"
        if channel not in IBM_CHANNELS:
            raise ConfigError(f"IBM_QUANTUM_CHANNEL must be one of {IBM_CHANNELS}, got {channel!r}")

        data_dir = get("QL_DATA_DIR")
        return cls(
            ibm_token=get("IBM_QUANTUM_TOKEN") or get("QISKIT_IBM_TOKEN"),
            ibm_instance=get("IBM_QUANTUM_INSTANCE"),
            ibm_channel=channel,
            ibm_backend=get("IBM_BACKEND"),
            ibm_max_shots_per_job=get_int("IBM_MAX_SHOTS_PER_JOB", 20000),
            ibm_max_total_shots=get_int("IBM_MAX_TOTAL_SHOTS", 200000),
            ibm_max_jobs=get_int("IBM_MAX_JOBS", 100),
            sim_noise=noise,
            sim_latency_seconds=latency,
            data_dir=Path(data_dir).expanduser() if data_dir else PROJECT_ROOT / ".data",
        )


def load_settings(env_file: Path | None = None) -> Settings:
    """Process environment wins over the .env file. A named --env-file must exist."""
    path = env_file if env_file is not None else DEFAULT_ENV_FILE
    if env_file is not None and not path.is_file():
        raise ConfigError(f"env file not found: {path}")
    merged: dict[str, str] = {}
    if path.is_file():
        merged.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    merged.update(os.environ)
    return Settings.from_env(merged)
