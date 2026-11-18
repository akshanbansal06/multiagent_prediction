import datetime as dt
import numpy as np
from src.agents.logreg_agent import LogRegAgent, DEFAULTS

D = dt.datetime(2020, 1, 1)
M = {"match_id": 1, "date": D, "season": "1920", "home": "A", "away": "B"}
R = {"home_goals": 3, "away_goals": 0, "outcome": 0}


def test_probabilities_are_valid_before_and_after_fitting():
    a = LogRegAgent(min_rows=20, refit_every=10)
    p = a.predict(M)
    assert np.isclose(p.sum(), 1.0) and (p > 0).all()
    rng = np.random.default_rng(0)
    for i in range(60):
        m = {"match_id": 100 + i, "date": D + dt.timedelta(days=i),
             "season": "1920", "home": f"T{i % 6}", "away": f"T{(i + 1) % 6}"}
        a.predict(m)
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        a.update(m, {"home_goals": hg, "away_goals": ag,
                     "outcome": 0 if hg > ag else 1 if hg == ag else 2})
    assert a.model is not None
    p = a.predict(M)
    assert np.isclose(p.sum(), 1.0) and (p > 0).all()


def test_features_do_not_include_the_match_being_predicted():
    a = LogRegAgent()
    before = a.features(M)
    a.predict(M)
    assert a.features(M) == before                 # predicting changes nothing
    assert before[:4] == list(DEFAULTS)            # no history yet
    a.update(M, R)
    later = dict(M, match_id=2, date=D + dt.timedelta(days=7))
    f = a.features(later)
    assert f[0] == 3.0 and f[1] == 0.0 and f[2] == 3.0   # home team's one past game
    assert f[4] == 0.0 and f[5] == 3.0 and f[6] == 0.0   # away team's one past game
