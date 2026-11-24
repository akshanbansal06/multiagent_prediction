"""Composite trust: four cheap signals per agent, each one running number (O(N) per match).

accuracy     recent RPS, with old matches forgotten           (low is good)
calibration  how much more confident than correct it has been (catches overconfidence)
shock        recent loss minus long-run loss                  (catches sudden change, e.g. sleepers)
validity     recent share of forecasts that passed validation  (catches broken output)

penalty_i = accuracy_i + lam_cal * calibration_i + lam_shock * shock_i
weight_i  is proportional to  validity_i ** 2 * exp(-eta * (penalty_i - min_j penalty_j))
Accuracy is an average, not a running total, so weights never collapse onto one agent."""
import numpy as np


class Composite:
    name = "composite"

    def __init__(self, n_agents, eta=100.0, lam_cal=1.0, lam_shock=2.0,
                 g_long=0.995, g_short=0.95, g_valid=0.98):
        self.eta, self.lam_cal, self.lam_shock = eta, lam_cal, lam_shock
        self.g_long, self.g_short, self.g_valid = g_long, g_short, g_valid
        self.acc = np.full(n_agents, 0.22)       # start everyone at a typical RPS
        self.short = np.full(n_agents, 0.22)
        self.cal = np.zeros(n_agents)
        self.valid = np.ones(n_agents)

    def components(self):
        shock = np.maximum(0.0, self.short - self.acc)
        cal = np.maximum(0.0, self.cal)          # only over-confidence is penalised
        penalty = self.acc + self.lam_cal * cal + self.lam_shock * shock
        return {"accuracy": self.acc, "calibration": cal, "shock": shock,
                "validity": self.valid, "penalty": penalty}

    def weights(self):
        c = self.components()
        w = self.valid ** 2 * np.exp(-self.eta * (c["penalty"] - c["penalty"].min()))
        return w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

    def update(self, losses):
        self.acc = self.g_long * self.acc + (1 - self.g_long) * losses
        self.short = self.g_short * self.short + (1 - self.g_short) * losses

    def observe(self, forecasts, valid, outcome):
        """Extra information the aggregator passes after each result."""
        self.valid = self.g_valid * self.valid + (1 - self.g_valid) * valid
        conf = np.where(valid, np.nanmax(np.nan_to_num(forecasts, nan=0.0), axis=1), 0.0)
        hit = np.where(valid, np.nan_to_num(forecasts, nan=0.0).argmax(axis=1) == outcome, 0.0)
        gap = np.where(valid, conf - hit, 0.0)
        self.cal = self.g_long * self.cal + (1 - self.g_long) * gap

    def explain(self, names):
        c, w = self.components(), self.weights()
        lines = []
        for i in np.argsort(-w):
            reasons = []
            if c["shock"][i] > 0.02:
                reasons.append("recent form much worse than usual")
            if c["calibration"][i] > 0.1:
                reasons.append("more confident than accurate")
            if c["validity"][i] < 0.9:
                reasons.append(f"only {c['validity'][i]:.0%} of recent forecasts valid")
            why = "; ".join(reasons) if reasons else "no warning signs"
            lines.append(f"{names[i]:<12} weight {w[i]:.2f}  recent RPS {c['accuracy'][i]:.3f}  ({why})")
        return "\n".join(lines)
