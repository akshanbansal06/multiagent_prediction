import numpy as np
from src.trust.validation import is_valid
from src.evaluation.metrics import rps_batch

INVALID_LOSS = 1.0          # an invalid forecast counts as the worst possible RPS


def run_trust(forecasts, outcomes, method):
    """forecasts: (n_matches, n_agents, 3). Walk forward: weight, combine, score, update.
    Returns combined forecasts (n_matches, 3) and the weights used (n_matches, n_agents)."""
    n, k, _ = forecasts.shape
    combined = np.zeros((n, 3))
    used = np.zeros((n, k))
    for t in range(n):
        valid = np.array([is_valid(forecasts[t, i]) for i in range(k)])
        w = method.weights() * valid                     # rejected agents get no say
        w = w / w.sum() if w.sum() > 0 else valid / valid.sum()
        combined[t] = w @ np.nan_to_num(forecasts[t])
        used[t] = w
        losses = np.full(k, INVALID_LOSS)
        losses[valid] = rps_batch(forecasts[t, valid], np.full(valid.sum(), outcomes[t]))
        method.update(losses)                            # learn only after the result
    return combined, used
