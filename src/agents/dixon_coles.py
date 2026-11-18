import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from src.agents.base_agent import BaseAgent

MAX_GOALS = 10


class DixonColesAgent(BaseAgent):
    """Time-weighted Poisson goals model with the Dixon-Coles low-score correction.
    Fitted in two stages: Poisson strengths first, then the correction rho given those."""
    name = "dixon_coles"

    def __init__(self, xi=0.0015, ridge=0.5, min_rows=380, refit_every=40):
        self.xi = xi                      # time decay per day
        self.ridge = ridge                # pulls strengths toward 0 (also fixes identifiability)
        self.min_rows = min_rows
        self.refit_every = refit_every
        self.rows = []                    # (home, away, home_goals, away_goals, date)
        self.counts = np.ones(3)
        self.fitted = False

    @staticmethod
    def _nll_grad(theta, hi, ai, x, y, w, n_teams, ridge):
        mu, h = theta[0], theta[1]
        att, dfn = theta[2:2 + n_teams], theta[2 + n_teams:]
        eta_h = mu + h + att[hi] - dfn[ai]
        eta_a = mu + att[ai] - dfn[hi]
        lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)
        nll = np.sum(w * (lam_h - x * eta_h + lam_a - y * eta_a))
        nll += ridge * (np.sum(att ** 2) + np.sum(dfn ** 2))
        gh, ga = w * (lam_h - x), w * (lam_a - y)
        g = np.zeros_like(theta)
        g[0] = gh.sum() + ga.sum()
        g[1] = gh.sum()
        g[2:2 + n_teams] = (np.bincount(hi, gh, n_teams) + np.bincount(ai, ga, n_teams)
                            + 2 * ridge * att)
        g[2 + n_teams:] = (-np.bincount(ai, gh, n_teams) - np.bincount(hi, ga, n_teams)
                           + 2 * ridge * dfn)
        return nll, g

    def _fit(self):
        teams = sorted({r[0] for r in self.rows} | {r[1] for r in self.rows})
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([self.idx[r[0]] for r in self.rows])
        ai = np.array([self.idx[r[1]] for r in self.rows])
        x = np.array([r[2] for r in self.rows], float)
        y = np.array([r[3] for r in self.rows], float)
        days = np.array([(self.rows[-1][4] - r[4]).days for r in self.rows], float)
        w = np.exp(-self.xi * days)
        theta0 = np.zeros(2 + 2 * n)
        theta0[0], theta0[1] = np.log(1.3), 0.25
        res = minimize(self._nll_grad, theta0, args=(hi, ai, x, y, w, n, self.ridge),
                       jac=True, method="L-BFGS-B")
        self.mu, self.h = res.x[0], res.x[1]
        self.att, self.dfn = res.x[2:2 + n], res.x[2 + n:]
        # stage 2: choose rho from the low-score matches, given the fitted rates
        lam_h = np.exp(self.mu + self.h + self.att[hi] - self.dfn[ai])
        lam_a = np.exp(self.mu + self.att[ai] - self.dfn[hi])
        low = (x <= 1) & (y <= 1)
        best, best_ll = 0.0, -np.inf
        for rho in np.linspace(-0.3, 0.3, 61):
            t = self._tau(x[low], y[low], lam_h[low], lam_a[low], rho)
            if (t <= 0).any():
                continue
            ll = np.sum(w[low] * np.log(t))
            if ll > best_ll:
                best, best_ll = rho, ll
        self.rho = best
        self.fitted = True

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        t = np.ones_like(lam, dtype=float)
        t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
        t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
        t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
        t = np.where((x == 1) & (y == 1), 1 - rho, t)
        return t

    def grid(self, match):
        """grid[i, j] = P(home scores i, away scores j)."""
        def get(team):
            i = self.idx.get(team)
            return (self.att[i], self.dfn[i]) if i is not None else (0.0, 0.0)
        a_h, d_h = get(match["home"])
        a_a, d_a = get(match["away"])
        lam = np.exp(self.mu + self.h + a_h - d_a)
        mu = np.exp(self.mu + a_a - d_h)
        k = np.arange(MAX_GOALS + 1)
        g = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))
        g[0, 0] *= 1 - lam * mu * self.rho
        g[0, 1] *= 1 + lam * self.rho
        g[1, 0] *= 1 + mu * self.rho
        g[1, 1] *= 1 - self.rho
        return g / g.sum()

    def predict(self, match):
        if not self.fitted:
            return self.counts / self.counts.sum()
        g = self.grid(match)
        p = np.array([np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()])
        return p / p.sum()

    def update(self, match, result):
        self.rows.append((match["home"], match["away"], result["home_goals"],
                          result["away_goals"], match["date"]))
        self.counts[result["outcome"]] += 1
        n = len(self.rows)
        if n >= self.min_rows and n % self.refit_every == 0:
            self._fit()
