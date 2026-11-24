"""Fault injectors. Each takes an (n, 3) array of honest probabilities
[home, draw, away] in match order and returns a corrupted copy.
INVALID rows are NaN, which the validation layer must catch."""
import numpy as np


def _norm(p):
    p = np.clip(p, 1e-6, None)
    return p / p.sum(axis=1, keepdims=True)


def noisy(p, rng, sigma=0.15):
    """Honest signal plus random noise."""
    return _norm(p + rng.normal(0, sigma, p.shape))


def biased(p, rng=None, shift=0.12):
    """Always favours the home side."""
    q = p.copy()
    q[:, 0] += shift
    q[:, 2] -= shift
    return _norm(q)


def flipped(p, rng=None):
    """Swaps home and away: confidently wrong about the winner."""
    return p[:, ::-1].copy()


def overconfident(p, rng=None, power=3.0):
    """Sharpens the forecast towards its favourite."""
    return _norm(p ** power)


def sleeper(p, rng=None, start=1500, fault=flipped):
    """Honest until match index `start`, then switches to `fault`."""
    q = p.copy()
    q[start:] = fault(p[start:], rng)
    return q


def invalid(p, rng, rate=0.1):
    """A fraction of rows are unusable (NaN)."""
    q = p.copy()
    bad = rng.random(len(p)) < rate
    q[bad] = np.nan
    return q


FAULTS = {
    "noisy": noisy,
    "biased": biased,
    "flipped": flipped,
    "overconfident": overconfident,
    "sleeper": sleeper,
    "invalid": invalid,
}
