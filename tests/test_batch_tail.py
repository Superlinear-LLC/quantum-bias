"""The stream `batch` event carries the trailing run of aggregated bits (UI streak across batches)."""
import asyncio

import numpy as np
import pytest

from fakes import ScriptedSim
from quantum_lite.app import create_app
from quantum_lite.config import Settings
from quantum_lite.sessions import tail_run


@pytest.mark.parametrize("bits,expected", [
    ([1], (1, 1)),
    ([0, 0, 0], (0, 3)),
    ([1, 1, 0], (0, 1)),
    ([0, 1, 1, 1], (1, 3)),
    ([1, 0, 1, 0, 0], (0, 2)),
])
def test_tail_run_is_the_last_bits_run_length(bits, expected):
    got = tail_run(np.array(bits, dtype=np.uint8))
    assert (got["tail_bit"], got["tail_run"]) == expected


def test_tail_run_of_empty_batch_raises():
    with pytest.raises(ValueError, match="empty batch"):
        tail_run(np.zeros(0, dtype=np.uint8))


def _tails_of(settings, batches, num_qubits, aggregation):
    """Run a live stream whose jobs measure the known bits; return (tail_bit, tail_run, shots) of each event."""
    async def main():
        app = create_app(settings, providers={"sim": ScriptedSim(settings, batches, when_empty="block")})
        mgr = app.state.mgr
        s = await mgr.create({"provider": "sim", "mode": "live", "num_qubits": num_qubits, "aggregation": aggregation,
                              "shots_per_job": 40, "interval_ms": 20, "auto_refill": False,
                              "refill_below": 0, "refill_shots": None,
                              "stream_seconds": 60, "stream_max_jobs": None})
        sub = mgr.bus.subscribe(s.id)
        await mgr.start_stream(s)
        got = []
        while len(got) < len(batches):
            ev, data = await asyncio.wait_for(sub.queue.get(), 10)
            if ev == "batch":
                got.append((data["tail_bit"], data["tail_run"], data["shots"]))
        await mgr.stop_stream(s)
        await mgr.shutdown()
        return got

    return asyncio.run(main())


def test_stream_batch_events_carry_the_exact_tail_of_each_batch(tmp_path):
    """Drive the stream with known bits: each event's tail must be that batch's own trailing run."""
    known = [
        [0] * 37 + [1, 1, 1],       # tail (1, 3)
        [1] * 40,                   # whole batch one run: (1, 40)
        [1] * 39 + [0],             # (0, 1)
        [1, 0] * 10 + [0] * 20,     # (0, 21)
    ]
    expected = [(1, 3), (1, 40), (0, 1), (0, 21)]
    got = _tails_of(Settings(data_dir=tmp_path / "d"), [np.array(k).reshape(-1, 1) for k in known], 1, "none")
    assert got == [(b, r, 40) for b, r in expected]


def test_the_tail_is_of_the_aggregated_bits_not_of_a_raw_qubit(tmp_path):
    """3 qubits, majority: raw qubit 0 is 1 in every shot (a run of 40), the aggregated bit is 0 in
    the last 5 shots. The event must say (0, 5); (1, 40) would be the raw first column."""
    agg = [1] * 35 + [0] * 5
    raw = np.stack([np.ones(40, dtype=np.uint8), np.array(agg, dtype=np.uint8), np.array(agg, dtype=np.uint8)], axis=1)
    assert (np.sum(raw, axis=1) >= 2).astype(np.uint8).tolist() == agg  # the fixture is a real majority
    got = _tails_of(Settings(data_dir=tmp_path / "d"), [raw], 3, "majority")
    assert got == [(0, 5, 40)]
