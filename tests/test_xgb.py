import datetime as dt
import numpy as np
from src.agents.xgb_agent import XGBAgent

D = dt.datetime(2020, 1, 1)


def test_probabilities_are_valid_before_and_after_fitting():
    a = XGBAgent(min_rows=30, refit_every=10, n_estimators=10)
    rng = np.random.default_rng(0)
    for i in range(80):
        m = {"match_id": i, "date": D + dt.timedelta(days=i), "season": "1920",
             "home": f"T{i % 6}", "away": f"T{(i + 1) % 6}"}
        p = a.predict(m)
        assert np.isclose(p.sum(), 1.0) and (p > 0).all()
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        a.update(m, {"home_goals": hg, "away_goals": ag,
                     "outcome": 0 if hg > ag else 1 if hg == ag else 2})
    assert a.model is not None
