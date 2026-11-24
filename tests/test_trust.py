import numpy as np
from src.trust.validation import is_valid
from src.trust.methods import EqualWeights, Hedge
from src.trust.aggregator import run_trust


def test_validation():
    assert is_valid([0.5, 0.3, 0.2])
    assert not is_valid([0.5, 0.3, 0.3])
    assert not is_valid([np.nan, 0.5, 0.5])
    assert not is_valid([1.2, -0.1, -0.1])


def test_equal_weights_average():
    f = np.array([[[0.6, 0.2, 0.2], [0.2, 0.2, 0.6]]])
    c, w = run_trust(f, np.array([0]), EqualWeights(2))
    assert np.allclose(c[0], [0.4, 0.2, 0.4])


def test_invalid_agent_is_ignored():
    f = np.array([[[0.6, 0.2, 0.2], [np.nan, np.nan, np.nan]]])
    c, w = run_trust(f, np.array([0]), EqualWeights(2))
    assert np.allclose(c[0], [0.6, 0.2, 0.2]) and w[0, 1] == 0


def test_hedge_moves_weight_to_the_better_agent():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.1); good[np.arange(n), y] = 0.8
    bad = good[:, ::-1].copy()
    f = np.stack([good, bad], axis=1)
    _, w = run_trust(f, y, Hedge(2, eta=1.0))
    assert w[-1, 0] > 0.95


def test_weights_sum_to_one():
    rng = np.random.default_rng(1)
    f = rng.dirichlet([2, 2, 2], (50, 3))
    _, w = run_trust(f, rng.integers(0, 3, 50), Hedge(3))
    assert np.allclose(w.sum(1), 1)


def test_discounted_hedge_reacts_faster_to_a_sleeper():
    from src.trust.methods import DiscountedHedge
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.1); good[np.arange(n), y] = 0.8
    ok = np.full((n, 3), 0.25); ok[np.arange(n), y] = 0.5
    sleeper = good.copy(); sleeper[1500:] = good[1500:, ::-1]
    f = np.stack([sleeper, ok], axis=1)
    _, w_h = run_trust(f, y, Hedge(2, eta=1.0))
    _, w_d = run_trust(f, y, DiscountedHedge(2, eta=1.0, gamma=0.98))
    assert w_d[1550, 0] < w_h[1550, 0]
