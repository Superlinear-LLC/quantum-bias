"""IBM shot/job ceilings, per process. Only IBM counts; the simulator is free."""
from __future__ import annotations

from quantum_lite.config import Settings
from quantum_lite.errors import AppError


class IbmBudget:
    def __init__(self, settings: Settings) -> None:
        self.max_shots_per_job = settings.ibm_max_shots_per_job
        self.max_total_shots = settings.ibm_max_total_shots
        self.max_jobs = settings.ibm_max_jobs
        self.used_shots = 0
        self.used_jobs = 0

    def check_job_size(self, shots: int) -> None:
        if shots > self.max_shots_per_job:
            raise AppError(
                400, "shots_exceed_job_ceiling",
                f"{shots} shots requested, IBM_MAX_SHOTS_PER_JOB is {self.max_shots_per_job}",
            )

    def reserve(self, shots: int) -> None:
        """Count a job before it is submitted. Not refunded on failure: IBM may have run it."""
        self.check_job_size(shots)
        if self.used_jobs + 1 > self.max_jobs:
            raise AppError(
                429, "job_ceiling",
                f"IBM_MAX_JOBS is {self.max_jobs}, {self.used_jobs} already submitted by this process",
            )
        if self.used_shots + shots > self.max_total_shots:
            raise AppError(
                429, "shot_ceiling",
                f"IBM_MAX_TOTAL_SHOTS is {self.max_total_shots}: {self.used_shots} used + {shots} requested",
            )
        self.used_shots += shots
        self.used_jobs += 1

    def snapshot(self) -> dict:
        return {
            "shots": {"used": self.used_shots, "ceiling": self.max_total_shots},
            "jobs": {"used": self.used_jobs, "ceiling": self.max_jobs},
            "max_shots_per_job": self.max_shots_per_job,
        }
