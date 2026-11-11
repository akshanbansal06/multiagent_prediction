import numpy as np

def rps(probs, outcome):
    """Ranked Probability Score for one match.
    probs: [home, draw, away]. outcome: 0 home win, 1 draw, 2 away win. Lower is better."""
    probs = np.asarray(probs, dtype=float)
    actual = np.zeros(3)
    actual[outcome] = 1.0
    diff = np.cumsum(probs) - np.cumsum(actual)
    return float(np.sum(diff[:-1] ** 2) / 2)

def rps_batch(probs, outcomes):
    """Vectorised RPS. probs: (n, 3). outcomes: (n,) ints. Returns (n,)."""
    P = np.asarray(probs, dtype=float)
    A = np.eye(3)[np.asarray(outcomes, dtype=int)]
    diff = np.cumsum(P, axis=1) - np.cumsum(A, axis=1)
    return (diff[:, :2] ** 2).sum(axis=1) / 2

def log_loss(probs, outcome, eps=1e-12):
    return float(-np.log(max(float(probs[outcome]), eps)))
