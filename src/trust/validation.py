import numpy as np


def is_valid(p, tol=1e-6):
    """A forecast is valid if it has 3 finite, non-negative numbers that sum to 1."""
    p = np.asarray(p, float)
    return (p.shape == (3,) and np.isfinite(p).all()
            and (p >= 0).all() and abs(p.sum() - 1) < tol)
