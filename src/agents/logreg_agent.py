from collections import defaultdict, deque
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.agents.base_agent import BaseAgent

DEFAULTS = (1.4, 1.4, 1.35, 7.0)    # goals for, goals against, points per game, rest days


class TeamHistory:
    """Each team's last `window` matches. Only add() changes it, and add() is
    called after the result is known, so features never include the current match."""

    def __init__(self, window=6):
        self.games = defaultdict(lambda: deque(maxlen=window))   # (gf, ga, points)
        self.last_date = {}

    def stats(self, team, date):
        g = self.games[team]
        if not g:
            return DEFAULTS
        gf = float(np.mean([x[0] for x in g]))
        ga = float(np.mean([x[1] for x in g]))
        ppg = float(np.mean([x[2] for x in g]))
        rest = min((date - self.last_date[team]).days, 14)
        return gf, ga, ppg, float(rest)

    def add(self, match, result):
        hg, ag = result["home_goals"], result["away_goals"]
        hp, ap = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
        self.games[match["home"]].append((hg, ag, hp))
        self.games[match["away"]].append((ag, hg, ap))
        self.last_date[match["home"]] = match["date"]
        self.last_date[match["away"]] = match["date"]


class LogRegAgent(BaseAgent):
    name = "logreg"

    def __init__(self, window=6, C=1.0, min_rows=300, refit_every=40):
        self.hist = TeamHistory(window)
        self.C = C
        self.min_rows = min_rows
        self.refit_every = refit_every
        self.X, self.y = [], []
        self.pending = {}
        self.model = None
        self.counts = np.ones(3)

    def features(self, match):
        h = self.hist.stats(match["home"], match["date"])
        a = self.hist.stats(match["away"], match["date"])
        return list(h) + list(a)

    def predict(self, match):
        x = self.features(match)
        self.pending[match["match_id"]] = x       # remember exactly what we saw before the result
        if self.model is None:
            return self.counts / self.counts.sum()
        return self.model.predict_proba([x])[0]

    def update(self, match, result):
        x = self.pending.pop(match["match_id"], None) or self.features(match)
        self.X.append(x)
        self.y.append(result["outcome"])
        self.counts[result["outcome"]] += 1
        self.hist.add(match, result)               # history changes only now
        n = len(self.y)
        if n >= self.min_rows and n % self.refit_every == 0:
            self.model = make_pipeline(StandardScaler(),
                                       LogisticRegression(C=self.C, max_iter=1000))
            self.model.fit(self.X, self.y)
