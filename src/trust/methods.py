"""Trust methods. Each keeps one number per agent and updates in O(N) per match."""
import numpy as np


class EqualWeights:
    name = "equal"

    def __init__(self, n_agents):
        self.n = n_agents

    def weights(self):
        return np.ones(self.n) / self.n

    def update(self, losses):
        pass


class Hedge:
    """Multiplicative weights: w_i is proportional to exp(-eta * cumulative loss_i)."""
    name = "hedge"

    def __init__(self, n_agents, eta=1.0):
        self.eta = eta
        self.cum = np.zeros(n_agents)

    def weights(self):
        z = -self.eta * (self.cum - self.cum.min())     # shift for numerical safety
        w = np.exp(z)
        return w / w.sum()

    def update(self, losses):
        self.cum += losses


class DiscountedHedge(Hedge):
    """Hedge with forgetting: old losses fade by gamma each match, so trust can recover or drop fast."""
    name = "discounted_hedge"

    def __init__(self, n_agents, eta=1.0, gamma=0.99):
        super().__init__(n_agents, eta)
        self.gamma = gamma

    def update(self, losses):
        self.cum = self.gamma * self.cum + losses
