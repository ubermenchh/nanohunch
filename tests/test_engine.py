"""engine.py parts 2 and 3: MLXBranchScorer on real MiniCPM5 weights (plan Phase 2 step 7). Slow."""

import json
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.slow

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = str(ROOT / "runs" / "models" / "minicpm5-2b-base-raw")
P0_ITEMS = ROOT / "data" / "raw" / "p0_items.jsonl"
GOLD_EVAL = ROOT / "data" / "skeleton" / "gold_eval.jsonl"


@pytest.fixture(scope="module")
def scorer():
    from engine import MLXBranchScorer

    return MLXBranchScorer(MODEL)


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _state_and_questions(tok, min_tokens=1000):
    """P0-4 setup: BoolQ passages joined until the prefix is >= min_tokens, 16 questions (8 BoolQ, 8 ARC)."""
    from fmt import Question
    from skeleton.fmt_ref import prefix_text

    rows = [json.loads(line) for line in P0_ITEMS.open()]
    boolq, arc = [r for r in rows if r["type"] == "noul"], [r for r in rows if r["type"] == "choice"]
    k = 1
    while len(tok.encode(prefix_text("\n\n".join(r["state"] for r in boolq[:k] * 4)))) < min_tokens:
        k += 1
    state = "\n\n".join(r["state"] for r in boolq[:k] * 4)
    qs = [Question(str(i), r["type"], r["question"], tuple(r["options"])) for i, r in enumerate(boolq[-8:] + arc[:8])]
    return state, qs


def test_head_path_matches_full_model(scorer):
    """Guard for Phase 2 step 6: backbone + gathered head rows == the model's own logits at those rows."""
    import mlx.core as mx

    from engine import label_logits

    ids = mx.array(
        scorer.tok.encode(
            "Science exam question.\n\nWhich gas do plants absorb?\nA. oxygen\nB. carbon dioxide\nAnswer:"
        )
    )[None]
    lab = [scorer.tok.encode(s, add_special_tokens=False)[0] for s in (" A", " B")]
    ours = np.array(label_logits(scorer._backbone(ids)[:, -1, :], scorer._head_weight, mx.array(lab))[0])
    full = np.array(scorer.model(ids)[0, -1].astype(mx.float32))[lab]
    # the full model rounds its logits to bf16 (about 3 significant digits); a wrong head, a missing
    # scale or an off-by-one position would be off by whole units, not by rounding
    assert np.allclose(ours, full, rtol=1e-2, atol=5e-2), (ours, full)
    assert int(ours.argmax()) == int(full.argmax())


def test_isolation_exact_and_repeatable(scorer):
    """A question scored alone == the same question among 16 (trim branching leaves no trace)."""
    from fmt import render

    state, qs = _state_and_questions(scorer.tok)
    r16 = render(scorer.tok, state, qs, n_perms=1, max_context=8192)
    r1 = render(scorer.tok, state, [qs[5]], n_perms=1, max_context=8192)
    assert len(r16.prefix_ids) >= 1000
    many, again, alone = scorer.score(r16), scorer.score(r16), scorer.score(r1)
    assert [b.question_id for b in many] == [q.id for q in qs]
    assert all(
        b.logits.dtype == np.float32 and b.logits.shape == (len(q.options),) for b, q in zip(many, qs, strict=True)
    )
    assert np.array_equal(many[5].logits, alone[0].logits)  # isolation diff exactly 0.0 (R8)
    assert all(np.array_equal(a.logits, b.logits) for a, b in zip(many, again, strict=True))


def test_chunked_prefill_matches_oneshot(scorer):
    from fmt import render

    state, qs = _state_and_questions(scorer.tok, min_tokens=3000)
    r = render(scorer.tok, state, qs[:2] + qs[8:10], n_perms=1, max_context=8192)
    assert len(r.prefix_ids) >= 3000
    chunked = scorer.score(r)
    old = scorer.prefill_chunk
    try:
        scorer.prefill_chunk = 10**9
        oneshot = scorer.score(r)
    finally:
        scorer.prefill_chunk = old
    diff = max(np.abs(_softmax(a.logits) - _softmax(b.logits)).max() for a, b in zip(chunked, oneshot, strict=True))
    assert diff <= 2e-2, diff


def test_reencode_bf16_close_to_branching(scorer):
    """Same bf16 model, full re-encode vs trim branching: different kernel shapes, so close, not equal."""
    from fmt import render

    state, qs = _state_and_questions(scorer.tok)
    r = render(scorer.tok, state, qs[:2] + qs[8:10], n_perms=1, max_context=8192)
    branched, reenc = scorer.score(r), scorer.score_reencode(r)
    assert [b.question_id for b in reenc] == [b.question_id for b in branched]
    assert all(b.logits.dtype == np.float32 for b in reenc)
    diff = max(np.abs(_softmax(a.logits) - _softmax(b.logits)).max() for a, b in zip(branched, reenc, strict=True))
    assert diff <= 5e-2, diff  # P0-4 measured 2.9e-2 in bf16 at a 1k state


def test_oracle_fp32_cpu():
    """R8 as amended at P0-4: on the CPU backend in fp32, branching == full re-encode within 1e-3."""
    import mlx.core as mx

    from engine import MLXBranchScorer
    from fmt import render

    mx.set_default_device(mx.cpu)
    try:
        oracle = MLXBranchScorer(MODEL, dtype="float32")
        state, qs = _state_and_questions(oracle.tok)
        r = render(oracle.tok, state, qs[:4] + qs[8:12], n_perms=1, max_context=8192)
        branched, reenc = oracle.score(r), oracle.score_reencode(r)
        diff = max(np.abs(_softmax(a.logits) - _softmax(b.logits)).max() for a, b in zip(branched, reenc, strict=True))
        print(f"cpu fp32 branch_vs_reencode = {diff:.2e}")
        assert diff <= 1e-3, diff
    finally:
        mx.set_default_device(mx.gpu)


@pytest.mark.skipif(not GOLD_EVAL.exists(), reason="run: uv run python -m skeleton.prep_gold")
def test_engine_matches_your_reader(scorer):
    """Your engine vs your Phase 1 reader (full re-encode) on 8 gold rows: the Phase 2 gate in miniature."""
    from fmt import Question, render
    from skeleton.b0_reader import read

    rows = [json.loads(line) for line in GOLD_EVAL.open()]
    worst = 0.0
    for i, row in enumerate(rows[:4] + rows[500:504]):  # 4 BoolQ, 4 ARC
        q = Question(str(i), row["type"], row["question"], tuple(row["options"]))
        (b,) = scorer.score(render(scorer.tok, row["state"], [q], n_perms=1, max_context=8192))
        worst = max(worst, float(np.abs(_softmax(b.logits) - np.array(read(scorer.model, scorer.tok, row))).max()))
    assert worst <= 2e-2, worst
