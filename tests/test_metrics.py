import numpy as np
from src.evaluation.metrics import rps, rps_batch, log_loss

def test_perfect_forecast_is_zero():
    assert rps([1, 0, 0], 0) == 0.0

def test_known_values():
    assert abs(rps([0.5, 0.3, 0.2], 0) - 0.145) < 1e-9
    assert abs(rps([0.5, 0.3, 0.2], 2) - 0.445) < 1e-9

def test_worst_forecast_is_one():
    assert abs(rps([0, 0, 1], 0) - 1.0) < 1e-9

def test_batch_matches_single():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    O = np.array([0, 2])
    assert np.allclose(rps_batch(P, O), [rps(P[0], 0), rps(P[1], 2)])

def test_log_loss():
    assert abs(log_loss([0.5, 0.3, 0.2], 0) - np.log(2)) < 1e-9
