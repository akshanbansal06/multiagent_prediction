import numpy as np
from src.faults.faults import FAULTS, noisy, flipped, sleeper, invalid, biased, overconfident


def honest(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([4, 2.5, 3], n)
    return p, rng


def test_valid_faults_return_probabilities():
    p, rng = honest()
    for name, f in FAULTS.items():
        if name == "invalid":
            continue
        q = f(p, rng)
        assert q.shape == p.shape
        assert np.allclose(q.sum(axis=1), 1) and (q > 0).all()


def test_flip_swaps_home_and_away_and_keeps_draw():
    p, rng = honest()
    q = flipped(p)
    assert np.allclose(q[:, 0], p[:, 2]) and np.allclose(q[:, 1], p[:, 1])


def test_sleeper_is_honest_then_faulty():
    p, rng = honest()
    q = sleeper(p, rng, start=1000)
    assert np.allclose(q[:1000], p[:1000])
    assert np.allclose(q[1000:], p[1000:][:, ::-1])


def test_invalid_marks_about_the_right_fraction_nan():
    p, rng = honest(5000)
    q = invalid(p, rng, rate=0.1)
    frac = np.isnan(q).any(axis=1).mean()
    assert 0.07 < frac < 0.13


def test_bias_raises_home_and_overconfidence_sharpens():
    p, rng = honest()
    assert biased(p)[:, 0].mean() > p[:, 0].mean()
    assert overconfident(p).max(axis=1).mean() > p.max(axis=1).mean()


def test_noise_leaves_the_original_untouched():
    p, rng = honest()
    before = p.copy()
    noisy(p, rng)
    assert np.array_equal(p, before)
