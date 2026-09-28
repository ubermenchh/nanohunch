"""calibrate.py part 1: temperature scaling (plan Phase 2 step 5). Pure numpy, no model."""

import numpy as np
import pytest


def _log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _sample(rng, n, k, true_t):
    """Targets drawn from softmax(z); the model reports log_softmax(true_t * z): overconfident if true_t > 1."""
    z = rng.normal(0, 1.5, (n, k))
    p = np.exp(_log_softmax(z))
    targets = [int(rng.choice(k, p=pi)) for pi in p]
    return list(_log_softmax(true_t * z)), targets


def test_temperature_recovery():
    from calibrate import fit_temperature

    # 20,000 items: at the plan's 5,000 the fitted T strayed up to 3.7% across seeds (measured 2026-09-25)
    x, t = _sample(np.random.default_rng(0), 20_000, 4, 2.5)
    assert abs(fit_temperature(x, t) - 2.5) / 2.5 < 0.05


def test_mixed_option_counts():
    from calibrate import fit_temperature

    rng = np.random.default_rng(1)
    x3, t3 = _sample(rng, 6_000, 3, 2.0)
    x5, t5 = _sample(rng, 6_000, 5, 2.0)
    assert abs(fit_temperature(x3 + x5, t3 + t5) - 2.0) / 2.0 < 0.05  # Choice items have 3 to 5 options


def test_temperature_hits_bounds_instead_of_diverging():
    """Degenerate inputs pin T to the search bound; fit-cal treats a T on a bound as a bug (Rule 6.4)."""
    from calibrate import T_MAX, T_MIN, fit_temperature

    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, (2_000, 4))
    sure = fit_temperature(list(_log_softmax(50 * z)), list(z.argmax(1)))  # always right: wants T -> 0
    noise = fit_temperature(list(_log_softmax(3 * z)), list(rng.integers(0, 4, 2_000)))  # unrelated: T -> inf
    assert (T_MIN, T_MAX) == (0.05, 20.0)
    assert sure == pytest.approx(T_MIN, rel=0.02) and noise == pytest.approx(T_MAX, rel=0.02)


def test_apply():
    from calibrate import Calibration, apply
    from fmt import FORMAT_VERSION

    cal = Calibration("rev", FORMAT_VERSION, {"choice:1": 2.0, "noul:1": 0.5}, {"choice": 1}, "eval_v1/cal@abc")
    pooled = np.log(np.array([0.7, 0.2, 0.1]))
    got = apply(cal, "choice", 1, pooled)
    want = np.exp(pooled / 2.0) / np.exp(pooled / 2.0).sum()
    assert np.allclose(got, want) and abs(got.sum() - 1) < 1e-12
    assert got.argmax() == pooled.argmax()  # temperature never changes the top-1 answer
    with pytest.raises(KeyError):
        apply(cal, "score", 1, pooled)  # no silent fallback to T = 1
    stale = Calibration("rev", "nanohunch-fmt-v0", {"choice:1": 2.0}, {"choice": 1}, "x")
    with pytest.raises(ValueError):
        apply(stale, "choice", 1, pooled)
