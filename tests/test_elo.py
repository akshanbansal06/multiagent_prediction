import numpy as np
from src.agents.elo import EloAgent

M = {"match_id": 0, "date": None, "season": "1516", "home": "A", "away": "B"}


def test_probabilities_are_valid():
    p = EloAgent().predict(M)
    assert p.shape == (3,)
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all()


def test_equal_teams_without_home_advantage_are_symmetric():
    p = EloAgent(home_adv=0).predict(M)
    assert np.isclose(p[0], p[2])


def test_home_advantage_favours_home():
    p = EloAgent(home_adv=100).predict(M)
    assert p[0] > p[2]


def test_win_raises_rating_and_probability():
    a = EloAgent()
    a.update(M, {"outcome": 0})
    assert a.ratings["A"] > 1500 and a.ratings["B"] < 1500
    assert a.predict(M)[0] > EloAgent().predict(M)[0]


def test_predict_does_not_change_ratings():
    a = EloAgent()
    a.update(M, {"outcome": 0})
    before = dict(a.ratings)
    a.predict(M)
    a.predict(M)
    assert a.ratings == before


def test_new_season_shrinks_ratings_toward_mean():
    a = EloAgent(carryover=0.5)
    a.update(M, {"outcome": 0})
    before = a.ratings["A"]
    a.predict(dict(M, season="1617"))
    assert abs(a.ratings["A"] - 1500) < abs(before - 1500)
