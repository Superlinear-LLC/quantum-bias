"""Pre-fetched measurement pools (buffered mode), persisted atomically as JSON.

A pool is keyed by (provider, implementation, n_qubits). Shots are consumed in order and
never reused: the cursor is persisted before `take` returns.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantum_lite.errors import AppError
from quantum_lite.providers.base import Measurement

_SAFE = re.compile(r"^[A-Za-z0-9_-]+$")


def pool_key(provider: str, implementation: str, n_qubits: int) -> str:
    for part in (provider, implementation):
        if not _SAFE.match(part):
            raise ValueError(f"unsafe pool key component {part!r}")
    return f"{provider}__{implementation}__{n_qubits}"


@dataclass(frozen=True)
class PoolBlock:
    """One job's bits with their origin: a block of a pool, a slice of one that `take` hands out, or a live job's."""

    job_id: str
    backend: str
    physical_qubits: tuple[int, ...]
    measured_at: str
    bits: np.ndarray  # (shots, n) uint8

    @classmethod
    def of(cls, m: Measurement) -> "PoolBlock":
        return cls(m.job_id, m.backend, m.physical_qubits, m.measured_at, m.bits)

    @property
    def shots(self) -> int:
        return int(self.bits.shape[0])


class Pool:
    def __init__(self, provider: str, implementation: str, n_qubits: int) -> None:
        self.provider = provider
        self.implementation = implementation
        self.n_qubits = n_qubits
        self.blocks: list[PoolBlock] = []
        self.cursor = 0  # shots consumed across all blocks, in order

    @property
    def key(self) -> str:
        return pool_key(self.provider, self.implementation, self.n_qubits)

    @property
    def total(self) -> int:
        return sum(b.shots for b in self.blocks)

    @property
    def available(self) -> int:
        return self.total - self.cursor

    def take(self, shots: int) -> list[PoolBlock]:
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        if shots > self.available:
            raise AppError(
                409,
                "pool_empty",
                f"pool {self.key} has {self.available} unconsumed shots, {shots} requested",
            )
        start, stop = self.cursor, self.cursor + shots
        pieces: list[PoolBlock] = []
        offset = 0  # pool-wide index of the first shot of the current block
        for block in self.blocks:
            lo, hi = max(start, offset), min(stop, offset + block.shots)
            if lo < hi:
                pieces.append(
                    PoolBlock(
                        block.job_id,
                        block.backend,
                        block.physical_qubits,
                        block.measured_at,
                        block.bits[lo - offset : hi - offset],
                    )
                )
            offset += block.shots
        self.cursor += shots
        return pieces

    def summary(self) -> dict:
        return {
            "key": self.key,
            "provider": self.provider,
            "implementation": self.implementation,
            "n_qubits": self.n_qubits,
            "total_shots": self.total,
            "consumed_shots": self.cursor,
            "available_shots": self.available,
            "blocks": [
                {"job_id": b.job_id, "backend": b.backend, "shots": b.shots, "measured_at": b.measured_at}
                for b in self.blocks
            ],
        }

    def to_json(self) -> dict:
        return {
            "provider": self.provider,
            "implementation": self.implementation,
            "n_qubits": self.n_qubits,
            "cursor": self.cursor,
            "blocks": [
                {
                    "job_id": b.job_id,
                    "backend": b.backend,
                    "physical_qubits": list(b.physical_qubits),
                    "measured_at": b.measured_at,
                    "shots": b.shots,
                    "bits": base64.b64encode(np.packbits(b.bits.ravel()).tobytes()).decode("ascii"),
                }
                for b in self.blocks
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Pool":
        pool = cls(data["provider"], data["implementation"], int(data["n_qubits"]))
        for raw in data["blocks"]:
            shots = int(raw["shots"])
            packed = np.frombuffer(base64.b64decode(raw["bits"], validate=True), dtype=np.uint8)
            expected_bytes = (shots * pool.n_qubits + 7) // 8
            # unpackbits(count=...) zero-pads short input, so the length must be checked first.
            if packed.size != expected_bytes:
                raise ValueError(
                    f"pool block {raw['job_id']} holds {packed.size} bytes of bits, expected {expected_bytes} "
                    f"for {shots} shots x {pool.n_qubits} qubits")
            flat = np.unpackbits(packed, count=shots * pool.n_qubits)
            pool.blocks.append(
                PoolBlock(
                    job_id=raw["job_id"],
                    backend=raw["backend"],
                    physical_qubits=tuple(raw["physical_qubits"]),
                    measured_at=raw["measured_at"],
                    bits=flat.reshape(shots, pool.n_qubits),
                )
            )
        pool.cursor = int(data["cursor"])
        if not (0 <= pool.cursor <= pool.total):
            raise ValueError(f"pool {pool.key} cursor {pool.cursor} outside 0..{pool.total}")
        return pool


def _lock_fd(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # blocks, retrying for ~10 s, then raises OSError
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


class PoolStore:
    """Pools on disk, shared by every process that uses the same data dir (server, CLI).

    The file is the truth. Every mutation (append, take, delete) runs under an exclusive lock
    on `<pools>/.lock`, re-reads the pool from disk first, changes it and writes it back, so a
    block added by another process is never overwritten and a cursor never moves backwards.
    Plain reads (`get`, `list`, `available`) refresh from disk when the file changed.
    """

    def __init__(self, data_dir: Path) -> None:
        self.dir = Path(data_dir) / "pools"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.Lock()
        self._pools: dict[str, Pool] = {}
        self._stamps: dict[str, tuple[int, int, int]] = {}
        for path in sorted(self.dir.glob("*.json")):  # corrupt file: fail at startup
            self._refresh(path.stem)

    # --- disk ------------------------------------------------------------------------
    @contextmanager
    def _locked(self):
        with self._mutex:
            fd = os.open(self.dir / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
            try:
                _lock_fd(fd)
                try:
                    yield
                finally:
                    _unlock_fd(fd)
            finally:
                os.close(fd)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def _read(self, key: str) -> Pool | None:
        path = self._path(key)
        try:
            text = path.read_text()
        except FileNotFoundError:
            return None
        pool = Pool.from_json(json.loads(text))
        if pool.key != key:
            raise ValueError(f"pool file {path.name} holds pool {pool.key}")
        return pool

    def _stamp(self, key: str) -> tuple[int, int, int] | None:
        try:
            st = self._path(key).stat()
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _refresh(self, key: str) -> Pool | None:
        """Bring the in-memory copy in line with the file (unlocked reads)."""
        stamp = self._stamp(key)
        if stamp is None:
            self._pools.pop(key, None)
            self._stamps.pop(key, None)
            return None
        if self._stamps.get(key) != stamp or key not in self._pools:
            pool = self._read(key)
            if pool is None:
                self._pools.pop(key, None)
                self._stamps.pop(key, None)
                return None
            self._pools[key] = pool
            self._stamps[key] = stamp
        return self._pools[key]

    def _save(self, pool: Pool) -> None:
        final = self._path(pool.key)
        tmp = self.dir / f"{pool.key}.json.{os.getpid()}.tmp"  # per-process: no shared tmp file
        try:
            with open(tmp, "w") as fh:
                json.dump(pool.to_json(), fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self._pools[pool.key] = pool
        stamp = self._stamp(pool.key)
        if stamp is not None:
            self._stamps[pool.key] = stamp

    # --- reads -----------------------------------------------------------------------
    def get(self, key: str) -> Pool | None:
        return self._refresh(key)

    def list(self) -> list[Pool]:
        on_disk = {p.stem for p in self.dir.glob("*.json")}
        for key in set(self._pools) - on_disk:
            self._refresh(key)  # deleted by another process
        pools = [self._refresh(key) for key in sorted(on_disk)]
        return [p for p in pools if p is not None]

    def available(self, key: str) -> int:
        pool = self._refresh(key)
        return 0 if pool is None else pool.available

    # --- mutations (locked, read-modify-write) ---------------------------------------
    def append(self, provider: str, implementation: str, n_qubits: int, block: PoolBlock) -> Pool:
        if block.bits.ndim != 2 or block.bits.shape[1] != n_qubits:
            raise ValueError(f"block bits shape {block.bits.shape} does not match n_qubits {n_qubits}")
        key = pool_key(provider, implementation, n_qubits)
        with self._locked():
            pool = self._read(key) or Pool(provider, implementation, n_qubits)
            pool.blocks.append(block)
            self._save(pool)
        return pool

    def take(self, key: str, shots: int) -> list[PoolBlock]:
        with self._locked():
            pool = self._read(key)
            if pool is None:
                self._pools.pop(key, None)
                raise AppError(409, "pool_empty", f"pool {key} does not exist (0 unconsumed shots, {shots} requested)")
            pieces = pool.take(shots)  # on failure nothing was written; the file is unchanged
            self._save(pool)
        return pieces

    def delete(self, key: str) -> None:
        with self._locked():
            if self._read(key) is None:
                self._pools.pop(key, None)
                raise AppError(404, "pool_not_found", f"no pool {key}")
            self._path(key).unlink()
            self._pools.pop(key, None)
            self._stamps.pop(key, None)
