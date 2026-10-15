import json
import os

import numpy as np
import pytest

from quantum_lite import pools as pools_mod
from quantum_lite.errors import AppError
from quantum_lite.pools import PoolBlock, PoolStore, pool_key


def block(job, shots, n=3, seed=0, start=0):
    rng = np.random.default_rng(seed)
    return PoolBlock(job, "aer_ideal", tuple(range(n)), "2026-01-01T00:00:00+00:00",
                     rng.integers(0, 2, size=(shots, n), dtype=np.uint8))


def store_with(tmp_path):
    store = PoolStore(tmp_path)
    store.append("sim", "aer_ideal", 3, block("j1", 5, seed=1))
    store.append("sim", "aer_ideal", 3, block("j2", 7, seed=2))
    return store


KEY = pool_key("sim", "aer_ideal", 3)


def test_round_trip_through_disk(tmp_path):
    store = store_with(tmp_path)
    reloaded = PoolStore(tmp_path)
    a, b = store.get(KEY), reloaded.get(KEY)
    assert b.total == a.total == 12 and b.cursor == 0
    for x, y in zip(a.blocks, b.blocks):
        assert x.job_id == y.job_id and x.backend == y.backend
        assert x.physical_qubits == y.physical_qubits and x.measured_at == y.measured_at
        assert np.array_equal(x.bits, y.bits)
    raw = json.loads((tmp_path / "pools" / f"{KEY}.json").read_text())
    assert isinstance(raw["blocks"][0]["bits"], str)  # base64 of packbits


def test_consumption_in_order_never_reused_and_cursor_persisted(tmp_path):
    store = store_with(tmp_path)
    pool = store.get(KEY)
    expected = np.concatenate([b.bits for b in pool.blocks])
    first = store.take(KEY, 4)
    second = store.take(KEY, 6)  # spans the j1/j2 boundary
    assert [p.job_id for p in first] == ["j1"]
    assert [p.job_id for p in second] == ["j1", "j2"]
    got = np.concatenate([p.bits for p in first + second])
    assert np.array_equal(got, expected[:10])
    # restart: cursor survives, the 10 consumed shots are not served again
    reloaded = PoolStore(tmp_path)
    assert reloaded.get(KEY).cursor == 10 and reloaded.get(KEY).available == 2
    third = reloaded.take(KEY, 2)
    assert np.array_equal(third[0].bits, expected[10:12])


def test_empty_and_missing_pool_are_409_not_fallbacks(tmp_path):
    store = store_with(tmp_path)
    store.take(KEY, 12)
    with pytest.raises(AppError) as e:
        store.take(KEY, 1)
    assert e.value.status == 409 and e.value.code == "pool_empty" and KEY in e.value.detail
    with pytest.raises(AppError) as e:
        store.take("sim__aer_ideal__9", 1)
    assert e.value.status == 409 and e.value.code == "pool_empty"


def test_request_larger_than_remaining_takes_nothing(tmp_path):
    store = store_with(tmp_path)
    with pytest.raises(AppError):
        store.take(KEY, 13)
    assert store.get(KEY).cursor == 0


def test_atomic_write_failure_leaves_old_file_and_cursor(tmp_path, monkeypatch):
    store = store_with(tmp_path)
    path = tmp_path / "pools" / f"{KEY}.json"
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pools_mod.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        store.take(KEY, 3)
    assert store.get(KEY).cursor == 0
    assert path.read_bytes() == before
    with pytest.raises(OSError):
        store.append("sim", "aer_ideal", 3, block("j3", 2))
    assert store.get(KEY).total == 12 and path.read_bytes() == before
    monkeypatch.undo()
    assert PoolStore(tmp_path).get(KEY).total == 12


def test_write_goes_through_tmp_and_rename(tmp_path, monkeypatch):
    calls = []
    real = os.replace
    monkeypatch.setattr(pools_mod.os, "replace", lambda a, b: (calls.append((str(a), str(b))), real(a, b)))
    PoolStore(tmp_path).append("sim", "aer_ideal", 3, block("j1", 2))
    assert calls and calls[0][0].endswith(".tmp") and str(os.getpid()) in calls[0][0] and calls[0][1].endswith(f"{KEY}.json")


def test_corrupt_pool_fails_startup(tmp_path):
    (tmp_path / "pools").mkdir(parents=True)
    (tmp_path / "pools" / "sim__aer_ideal__3.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        PoolStore(tmp_path)


def test_delete(tmp_path):
    store = store_with(tmp_path)
    store.delete(KEY)
    assert store.get(KEY) is None and not (tmp_path / "pools" / f"{KEY}.json").exists()
    with pytest.raises(AppError) as e:
        store.delete(KEY)
    assert e.value.status == 404


def test_second_store_append_survives_first_stores_take(tmp_path):
    """B2/D1/S1: a CLI prefetch appends while a server store holds an older in-memory copy."""
    server = PoolStore(tmp_path)
    server.append("sim", "aer_ideal", 3, block("j1", 100, seed=1))
    server.take(KEY, 50)
    cli = PoolStore(tmp_path)  # loaded after the server consumed 50
    cli.append("sim", "aer_ideal", 3, block("j2", 5000, seed=2))
    server.take(KEY, 1)  # the server still holds the old copy
    on_disk = PoolStore(tmp_path).get(KEY)
    assert [b.job_id for b in on_disk.blocks] == ["j1", "j2"]
    assert on_disk.total == 5100 and on_disk.cursor == 51


def test_stale_store_does_not_roll_cursor_back_or_reuse_shots(tmp_path):
    server = PoolStore(tmp_path)
    server.append("sim", "aer_ideal", 3, block("j1", 100, seed=1))
    cli = PoolStore(tmp_path)  # loaded with cursor 0
    first = server.take(KEY, 50)
    cli.append("sim", "aer_ideal", 3, block("j2", 10, seed=2))  # must not write cursor 0
    assert PoolStore(tmp_path).get(KEY).cursor == 50
    second = cli.take(KEY, 5)  # the cli copy was stale; it must serve shot 50 onwards
    expected = server.get(KEY).blocks[0].bits
    assert np.array_equal(second[0].bits, expected[50:55])
    assert not np.array_equal(first[0].bits[:5], second[0].bits)
    assert server.get(KEY).cursor == 55 and server.available(KEY) == 55


def test_reads_see_other_process_changes_and_deletes(tmp_path):
    a, b = PoolStore(tmp_path), PoolStore(tmp_path)
    a.append("sim", "aer_ideal", 3, block("j1", 10))
    assert b.available(KEY) == 10 and [p.key for p in b.list()] == [KEY]
    a.delete(KEY)
    assert b.get(KEY) is None and b.list() == []
    with pytest.raises(AppError) as e:
        b.take(KEY, 1)
    assert e.value.code == "pool_empty"


def test_concurrent_processes_never_hand_out_the_same_shot(tmp_path):
    import subprocess
    import sys
    import textwrap

    store = PoolStore(tmp_path)
    store.append("sim", "aer_ideal", 3, block("j1", 120, seed=3))
    script = textwrap.dedent(
        """
        import sys, json
        from pathlib import Path
        from quantum_lite.pools import PoolStore
        store = PoolStore(Path(sys.argv[1]))
        got = []
        for _ in range(20):
            got.append(store.take(sys.argv[2], 1)[0].bits.tolist())
        print(json.dumps(got))
        """
    )
    procs = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path), KEY], stdout=subprocess.PIPE, text=True)
             for _ in range(4)]
    outs = [json.loads(p.communicate()[0]) for p in procs]
    assert all(p.returncode == 0 for p in procs)
    assert PoolStore(tmp_path).get(KEY).cursor == 80
    served = [tuple(x[0]) for out in outs for x in out]
    expected = [tuple(r) for r in store.get(KEY).blocks[0].bits.tolist()][:80]
    assert sorted(served) == sorted(expected)  # 80 distinct slots handed out exactly once
