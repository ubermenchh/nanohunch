"""calibrate.py parts 2 and 3: metrics and the paired bootstrap (plan Phase 3 step 4). Pure numpy."""

import json
import pathlib

import numpy as np
import pytest

SKELETON = pathlib.Path(__file__).resolve().parent.parent / "reports" / "skeleton" / "b0_fwd.json"


def _log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


# ---- part 2: metrics ------------------------------------------------------------------------------


def test_ece_calibrated_sampler_near_zero():
    from calibrate import ece

    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 100_000)
    correct = rng.random(100_000) < conf
    assert ece(conf, correct, scheme="width") < 0.01
    assert ece(conf, correct, scheme="mass") < 0.01


def test_ece_known_values_and_last_bin():
    from calibrate import ece

    assert ece(np.array([1.0]), np.array([1])) == 0.0
    assert ece(np.array([1.0]), np.array([0])) == 1.0  # conf == 1.0 lands in the last bin, no IndexError
    assert ece(np.full(10, 0.9), np.array([1] * 8 + [0] * 2)) == pytest.approx(0.1)


def test_mass_bins_on_calibrated_groups():
    from calibrate import ece

    # 900 items at 0.95 and 100 at 0.55, each group perfectly calibrated. Mistakes are spread evenly
    # (1 in 20, 9 in 20): mass bins split tied confidences by position, so bunching the mistakes at
    # the end of a group would make one bin look miscalibrated.
    conf = np.r_[np.full(900, 0.95), np.full(100, 0.55)]
    correct = np.r_[np.tile([1] * 19 + [0], 45), np.tile([1] * 11 + [0] * 9, 5)]
    assert ece(conf, correct, scheme="width") == pytest.approx(0.0, abs=1e-12)
    assert ece(conf, correct, bins=10, scheme="mass") == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(not SKELETON.exists(), reason="run the Phase 1 B0 eval first")
def test_matches_skeleton_reference():
    """calibrate.accuracy and ece(width, 15) must equal skeleton/tiny_eval on the same inputs (plan Phase 1)."""
    from calibrate import accuracy, ece
    from skeleton.tiny_eval import accuracy as ref_acc
    from skeleton.tiny_eval import ece15

    rows = json.loads(SKELETON.read_text())
    probs = [np.array(r["probs"]) for r in rows]
    gold = [r["gold_index"] for r in rows]
    conf = np.array([p.max() for p in probs])
    correct = np.array([int(p.argmax()) == g for p, g in zip(probs, gold, strict=True)])
    assert accuracy(probs, gold) == ref_acc(probs, gold)
    assert ece(conf, correct, bins=15, scheme="width") == pytest.approx(ece15(conf, correct), abs=1e-12)


def test_nll_brier_accuracy():
    from calibrate import accuracy, brier, nll

    probs = [np.array([0.7, 0.2, 0.1]), np.array([0.5, 0.5])]
    targets = [0, 1]
    assert accuracy(probs, targets) == 0.5  # ties go to the first index, as np.argmax
    assert nll(probs, targets) == pytest.approx(-(np.log(0.7) + np.log(0.5)) / 2)
    assert brier(probs, targets) == pytest.approx(((0.09 + 0.04 + 0.01) + (0.25 + 0.25)) / 2)
    assert np.isfinite(nll([np.array([1.0, 0.0])], [1]))  # log(0) is clipped, not -inf


def test_flip_rate():
    from calibrate import flip_rate

    assert flip_rate([0, 1, 2, 1], [0, 1, 2, 1]) == 0.0
    assert flip_rate([0, 1], [1, 1]) == 0.5
    with pytest.raises(ValueError):
        flip_rate([0, 1], [0])


def test_temperature_recovery_end_to_end():
    from calibrate import ece, fit_temperature

    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, (20_000, 4))
    targets = [int(rng.choice(4, p=p)) for p in np.exp(_log_softmax(z))]
    logits = _log_softmax(2 * z)  # twice too confident
    temp = fit_temperature(list(logits), targets)
    assert 1.8 <= temp <= 2.2

    def top1_ece(lp):
        p = np.exp(lp)
        return ece(p.max(1), p.argmax(1) == np.array(targets), scheme="width")

    after = top1_ece(_log_softmax(logits / temp))
    assert after < 0.02 and after < top1_ece(logits)


# ---- part 3: paired bootstrap ---------------------------------------------------------------------


def test_bootstrap_self_zero():
    from calibrate import paired_bootstrap

    a = np.random.default_rng(1).integers(0, 2, 500)
    assert paired_bootstrap(a, a) == (0.0, 0.0, 0.0)
    assert paired_bootstrap(a, a, groups=np.repeat(np.arange(100), 5)) == (0.0, 0.0, 0.0)


def test_bootstrap_detects_real_gap_and_is_seeded():
    from calibrate import paired_bootstrap

    rng = np.random.default_rng(2)
    b = rng.random(3500) < 0.76
    a = b | (rng.random(3500) < 0.10)  # A fixes about 10% of B's mistakes and breaks nothing
    delta, lo, hi = paired_bootstrap(a.astype(int), b.astype(int), n=2000)
    assert delta == pytest.approx(a.mean() - b.mean()) and 0 < lo < delta < hi
    assert paired_bootstrap(a.astype(int), b.astype(int), n=2000) == (delta, lo, hi)
    assert all(isinstance(x, float) for x in (delta, lo, hi))


def test_group_bootstrap_is_wider():
    """Correlated items (5 decisions per state) make item resampling overconfident; groups fix it."""
    from calibrate import paired_bootstrap

    rng = np.random.default_rng(3)
    hard = np.repeat(rng.random(400) < 0.4, 5)  # whole states are hard for both models
    wins = np.repeat(rng.random(400) < 0.3, 5)  # A beats B on whole states, not single items
    b = np.where(hard, rng.random(2000) < 0.2, rng.random(2000) < 0.9)
    a = b | (wins & (rng.random(2000) < 0.8))
    groups = np.repeat(np.arange(400), 5)  # measured ratio of CI widths: 1.72
    _, lo_i, hi_i = paired_bootstrap(a.astype(int), b.astype(int), n=2000)
    _, lo_g, hi_g = paired_bootstrap(a.astype(int), b.astype(int), n=2000, groups=groups)
    assert (hi_g - lo_g) > 1.3 * (hi_i - lo_i)
