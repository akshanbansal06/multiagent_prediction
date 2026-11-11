import numpy as np

def odds_to_probs(odds):
    """odds: (n, 3) decimal odds for home, draw, away.
    Returns probabilities with the bookmaker margin (overround) removed."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)
