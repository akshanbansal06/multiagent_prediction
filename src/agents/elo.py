import math
import numpy as np
from src.agents.base_agent import BaseAgent


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class EloAgent(BaseAgent):
    name = "elo"

    def __init__(self, k=20.0, home_adv=65.0, cut=0.6, carryover=0.85,
                 scale=400.0, init=1500.0):
        self.k = k
        self.home_adv = home_adv
        self.cut = cut
        self.carryover = carryover
        self.scale = scale
        self.init = init
        self.ratings = {}
        self.season = None

    def _rating(self, team):
        return self.ratings.get(team, self.init)

    def _sync_season(self, match):
        """At a season change, shrink every rating toward the mean. Safe to call repeatedly."""
        season = match["season"]
        if self.season is not None and season != self.season:
            for team in self.ratings:
                self.ratings[team] = self.init + self.carryover * (self.ratings[team] - self.init)
        self.season = season

    def _diff(self, match):
        return self._rating(match["home"]) + self.home_adv - self._rating(match["away"])

    def predict(self, match):
        self._sync_season(match)
        z = self._diff(match) / self.scale * math.log(10)
        p_home = _sigmoid(z - self.cut)
        p_away = _sigmoid(-self.cut - z)
        p_draw = 1.0 - p_home - p_away
        return np.array([p_home, p_draw, p_away])

    def update(self, match, result):
        self._sync_season(match)
        diff = self._diff(match)
        expected = 1.0 / (1.0 + 10.0 ** (-diff / self.scale))
        actual = {0: 1.0, 1: 0.5, 2: 0.0}[result["outcome"]]
        delta = self.k * (actual - expected)
        self.ratings[match["home"]] = self._rating(match["home"]) + delta
        self.ratings[match["away"]] = self._rating(match["away"]) - delta
