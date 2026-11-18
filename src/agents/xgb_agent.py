from xgboost import XGBClassifier
from src.agents.logreg_agent import LogRegAgent


class XGBAgent(LogRegAgent):
    name = "xgboost"

    def __init__(self, window=6, min_rows=300, refit_every=40,
                 n_estimators=150, max_depth=2, learning_rate=0.05):
        super().__init__(window=window, min_rows=min_rows, refit_every=refit_every)
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=5.0,
                           objective="multi:softprob", random_state=0, n_jobs=1)

    def update(self, match, result):
        x = self.pending.pop(match["match_id"], None) or self.features(match)
        self.X.append(x)
        self.y.append(result["outcome"])
        self.counts[result["outcome"]] += 1
        self.hist.add(match, result)               # history changes only now
        n = len(self.y)
        if n >= self.min_rows and n % self.refit_every == 0:
            self.model = XGBClassifier(**self.params)
            self.model.fit(self.X, self.y)
