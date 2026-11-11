import numpy as np
from src.agents.base_agent import BaseAgent

class NaiveAgent(BaseAgent):
    name = "naive"

    def __init__(self):
        self.counts = np.ones(3)          # one pseudo-match per outcome to start

    def predict(self, match):
        return self.counts / self.counts.sum()

    def update(self, match, result):
        self.counts[result["outcome"]] += 1
