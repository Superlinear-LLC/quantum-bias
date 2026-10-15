import math

import numpy as np

from quantum_lite.stats import HitCounter, Stats, WINDOW, bias_pct, z_score, BitCounter


def test_zero_bits():
    assert bias_pct(0, 0) == 0.0
    assert z_score(0, 0) == 0.0
    snap = BitCounter().snapshot()
    assert snap["n"] == 0 and snap["bias_pct"] == 0.0 and snap["z"] == 0.0
    assert snap["window"]["n"] == 0


def test_hand_computed_bias_and_z():
    # N=4, k=3: bias = (3-2)/2*100 = 50 ; z = (3-2)/(sqrt(4)/2) = 1
    assert bias_pct(3, 4) == 50.0
    assert z_score(3, 4) == 1.0
    # N=100, k=60: bias = 10/50*100 = 20 ; z = 10/5 = 2
    assert bias_pct(60, 100) == 20.0
    assert z_score(60, 100) == 2.0
    # negative side
    assert bias_pct(40, 100) == -20.0
    assert z_score(40, 100) == -2.0


def test_sliding_window_uses_last_1000_bits():
    c = BitCounter()
    c.add(np.zeros(500, dtype=np.uint8))
    c.add(np.ones(1000, dtype=np.uint8))
    s = c.snapshot()
    assert s["n"] == 1500 and s["ones"] == 1000
    assert s["window"]["n"] == WINDOW == 1000
    assert s["window"]["ones"] == 1000
    assert s["window"]["bias_pct"] == 100.0
    assert math.isclose(s["window"]["z"], 500 / (math.sqrt(1000) / 2))
    assert math.isclose(s["bias_pct"], (1000 - 750) / 750 * 100)


def test_per_qubit_and_aggregated_counts():
    st = Stats(3)
    m = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    st.add_block(m, m[:, 0])
    snap = st.snapshot()
    assert snap["shots"] == 4
    assert snap["raw"]["n"] == 12 and snap["raw"]["ones"] == 6
    assert snap["aggregated"]["n"] == 4 and snap["aggregated"]["ones"] == 3
    assert snap["per_qubit_ones_fraction"] == [0.75, 0.25, 0.5]


def test_per_qubit_fraction_is_null_without_shots():
    assert Stats(2).snapshot()["per_qubit_ones_fraction"] == [None, None]


def test_hit_counter_streaks():
    h = HitCounter()
    assert h.snapshot()["hit_rate"] is None
    for hit in (True, True, False, True):
        h.add(hit)
    s = h.snapshot()
    assert (s["trials"], s["hits"], s["streak"], s["best_streak"]) == (4, 3, 1, 2)
    assert s["hit_rate"] == 0.75
    assert s["z_hits"] == 1.0  # (3-2)/(sqrt(4)/2)
