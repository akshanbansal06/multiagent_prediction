import numpy as np

class BaseAgent:
    name = "base"

    def predict(self, match) -> np.ndarray:
        """Return [home, draw, away] probabilities summing to 1."""
        raise NotImplementedError

    def update(self, match, result) -> None:
        """Learn from a finished match. Called only after predict."""
        pass
