import numpy as np
from src.evaluation.benchmarks import odds_to_probs

def test_odds_to_probs_sums_to_one():
    p = odds_to_probs([[2.0, 4.0, 4.0], [1.5, 4.0, 7.0]])
    assert np.allclose(p.sum(axis=1), 1.0)

def test_fair_odds_unchanged():
    p = odds_to_probs([[2.0, 4.0, 4.0]])
    assert np.allclose(p[0], [0.5, 0.25, 0.25])
