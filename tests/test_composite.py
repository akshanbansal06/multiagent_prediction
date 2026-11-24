import numpy as np
from src.trust.composite import Composite
from src.trust.aggregator import run_trust


def make(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.15); good[np.arange(n), y] = 0.7
    ok = np.full((n, 3), 0.25); ok[np.arange(n), y] = 0.5
    return y, good, ok


def test_weights_valid_and_cost_is_linear_state():
    y, good, ok = make()
    m = Composite(2)
    _, w = run_trust(np.stack([good, ok], 1), y, m)
    assert np.allclose(w.sum(1), 1) and (w >= 0).all()
    assert m.acc.shape == (2,)                      # one number per agent per signal


def test_keeps_two_similar_good_agents_both_weighted():
    y, good, _ = make()
    other = good.copy()
    _, w = run_trust(np.stack([good, other], 1), y, Composite(2))
    assert 0.3 < w[-1, 0] < 0.7                     # does not collapse onto one


def test_sleeper_dropped_quickly():
    y, good, ok = make()
    sleeper = good.copy(); sleeper[1500:] = good[1500:, ::-1]
    _, w = run_trust(np.stack([sleeper, ok], 1), y, Composite(2))
    assert w[1520, 0] < 0.05


def test_overconfidence_is_penalised():
    y, good, _ = make()
    rng = np.random.default_rng(3)
    wrong = rng.random(len(y)) < 0.4                 # right only 60% of the time...
    oc = np.full((len(y), 3), 0.025); oc[np.arange(len(y)), y] = 0.95
    oc[wrong] = oc[wrong][:, ::-1]                   # ...but always 95% sure
    m = Composite(2)
    run_trust(np.stack([good, oc], 1), y, m)
    assert m.components()["calibration"][1] > m.components()["calibration"][0]


def test_explain_mentions_every_agent():
    y, good, ok = make(300)
    m = Composite(2)
    run_trust(np.stack([good, ok], 1), y, m)
    text = m.explain(["good", "ok"])
    assert "good" in text and "ok" in text
