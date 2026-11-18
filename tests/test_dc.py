import datetime as dt
import numpy as np
from scipy.optimize import check_grad
from src.agents.dixon_coles import DixonColesAgent

D = dt.datetime(2020, 1, 1)


def synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    att = rng.normal(0, 0.25, 8)
    dfn = rng.normal(0, 0.25, 8)
    out = []
    for i in range(n):
        h, a = rng.choice(8, 2, replace=False)
        lam = np.exp(0.2 + 0.25 + att[h] - dfn[a])
        mu = np.exp(0.2 + att[a] - dfn[h])
        hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
        out.append(({"match_id": i, "date": D + dt.timedelta(days=i // 4), "season": "x",
                     "home": teams[h], "away": teams[a]},
                    {"home_goals": hg, "away_goals": ag,
                     "outcome": 0 if hg > ag else 1 if hg == ag else 2}))
    return out


def test_gradient_is_correct():
    rng = np.random.default_rng(1)
    n = 5
    hi, ai = rng.integers(0, n, 50), rng.integers(0, n, 50)
    x, y = rng.poisson(1.4, 50).astype(float), rng.poisson(1.1, 50).astype(float)
    w = rng.random(50)
    theta = rng.normal(0, 0.1, 2 + 2 * n)
    f = lambda t: DixonColesAgent._nll_grad(t, hi, ai, x, y, w, n, 0.5)[0]
    g = lambda t: DixonColesAgent._nll_grad(t, hi, ai, x, y, w, n, 0.5)[1]
    assert check_grad(f, g, theta) < 1e-4


def test_probabilities_valid_and_grid_sums_to_one():
    a = DixonColesAgent(min_rows=200, refit_every=100)
    for m, r in synthetic():
        p = a.predict(m)
        assert np.isclose(p.sum(), 1.0) and (p > 0).all()
        a.update(m, r)
    assert a.fitted
    assert np.isclose(a.grid(m).sum(), 1.0)


def test_learns_that_home_advantage_is_positive():
    a = DixonColesAgent(min_rows=200, refit_every=100)
    for m, r in synthetic():
        a.predict(m)
        a.update(m, r)
    assert a.h > 0.1
