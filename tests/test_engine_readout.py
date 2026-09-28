"""engine.py part 1: label readout, pooling, answers (plan Phase 2 step 4; ADR-0002). No model needed."""

import numpy as np
import pytest

SOFTMAX_2_05_M1 = [0.786, 0.175, 0.039]  # softmax([2.0, 0.5, -1.0])


def test_label_logits_matches_full_product():
    import mlx.core as mx

    from engine import label_logits

    mx.random.seed(0)
    hidden = mx.random.normal((4, 8)).astype(mx.bfloat16)
    head = mx.random.normal((50, 8)).astype(mx.bfloat16)
    ids = mx.array([3, 17, 42, 9])
    z = label_logits(hidden, head, ids)
    assert z.shape == (4, 4) and z.dtype == mx.float32
    full = hidden.astype(mx.float32) @ head.astype(mx.float32).T  # what we must never build at V = 130k
    assert np.allclose(np.array(z), np.array(full[:, ids]), atol=1e-5)


def test_label_logits_rejects_quantized_head():
    import mlx.core as mx

    from engine import label_logits

    packed = mx.zeros((50, 1), dtype=mx.uint32)  # how mlx stores quantized weights
    with pytest.raises(TypeError):
        label_logits(mx.zeros((1, 8)), packed, mx.array([0, 1]))


def test_pool_maps_to_canonical():
    from engine import BranchLogits, pool

    a = BranchLogits("q1", (0, 1, 2), np.array([2.0, 0.5, -1.0], dtype=np.float32))
    b = BranchLogits("q1", (2, 1, 0), np.array([-1.0, 0.5, 2.0], dtype=np.float32))
    assert np.allclose(np.exp(pool([a, b])["q1"]), SOFTMAX_2_05_M1, atol=1e-3)
    # (1, 2, 0) is not its own inverse: display d shows canonical perm[d]. An inverted remap
    # would give [0.039, 0.786, 0.175] here, and the reversal above cannot tell the difference.
    c = BranchLogits("q1", (1, 2, 0), np.array([0.5, -1.0, 2.0], dtype=np.float32))
    assert np.allclose(np.exp(pool([c])["q1"]), SOFTMAX_2_05_M1, atol=1e-3)


def test_pool_is_log_linear_and_keyed_by_question():
    from engine import BranchLogits, pool

    lp1, lp2 = np.log([0.6, 0.3, 0.1]), np.log([0.2, 0.5, 0.3])
    out = pool(
        [
            BranchLogits("q1", (0, 1, 2), lp1.astype(np.float32)),
            BranchLogits("q1", (0, 1, 2), lp2.astype(np.float32)),
            BranchLogits("q2", (0, 1), np.array([0.0, 0.0], dtype=np.float32)),
        ]
    )
    geo = np.exp((lp1 + lp2) / 2)
    assert np.allclose(np.exp(out["q1"]), geo / geo.sum(), atol=1e-6)  # geometric mean, renormalized
    assert np.allclose(np.exp(out["q2"]), [0.5, 0.5]) and set(out) == {"q1", "q2"}
    assert abs(np.exp(out["q1"]).sum() - 1) < 1e-9


def test_score_expected_value():
    from engine import to_answer
    from fmt import Question

    q = Question("s", "score", "How bad?", ("low", "mid", "high"), (0, 1, 2))
    a = to_answer(q, np.array([0.2, 0.3, 0.5]))
    assert a.expected == pytest.approx(1.3) and a.confidence == pytest.approx(0.5) and a.qtype == "score"


def test_noul_and_choice_answers():
    from engine import to_answer
    from fmt import Question

    a = to_answer(Question("n", "noul", "Sure?", ("yes", "no")), np.array([0.7, 0.3]))
    assert a.expected == pytest.approx(0.7) and a.confidence == pytest.approx(0.7)
    assert 0 < a.entropy_norm < 1
    c = to_answer(Question("c", "choice", "Pick", ("x", "y", "z", "w")), np.full(4, 0.25))
    assert c.expected is None and c.entropy_norm == pytest.approx(1.0) and c.question_id == "c"
    sure = to_answer(Question("c", "choice", "Pick", ("x", "y")), np.array([1.0, 0.0]))
    assert sure.entropy_norm == pytest.approx(0.0)  # no NaN from 0 * log 0
