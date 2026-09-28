"""fmt.py: nanohunch-fmt-v1 types, label vocabulary, permutations (plan Phase 2; ADR-0003 + Amendment 1).

Part 1: dataclasses, errors, FORMAT_VERSION, label_vocab, permutations_for.
Part 2: render (split tokenization, Score space id, perms, validation, context limit).
"""

import dataclasses
import json
import pathlib

import pytest

from skeleton.fmt_ref import branch_text, prefix_text

GOLD_EVAL = pathlib.Path(__file__).resolve().parent.parent / "data" / "skeleton" / "gold_eval.jsonl"


class FakeTok:
    """Encodes every label string to one id, except the ones listed in `split`."""

    def __init__(self, split=()):
        self.split = set(split)

    def encode(self, s, add_special_tokens=True):
        return [5, 6] if s in self.split else [100 + sum(map(ord, s))]


# ---- part 1 --------------------------------------------------------------------------------------


def test_format_version_and_frozen_types():
    from fmt import FORMAT_VERSION, Question

    assert FORMAT_VERSION == "nanohunch-fmt-v1"
    q = Question("q1", "choice", "Q?", ("x", "y"))
    assert q.values == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.text = "changed"
    assert not hasattr(q, "__dict__")  # slots=True


def test_errors_are_value_errors():
    from fmt import LabelNotSingleToken, StateTooLong, TooManyOptions

    assert all(issubclass(e, ValueError) for e in (TooManyOptions, StateTooLong, LabelNotSingleToken))


def test_label_vocab_minicpm(tok):
    from fmt import label_vocab

    assert label_vocab(tok, "choice", 3) == tuple(
        tok.encode(s, add_special_tokens=False)[0] for s in [" A", " B", " C"]
    )
    assert label_vocab(tok, "noul", 2) == (8225, 768)  # " yes", " no" (P0-1)
    # Amendment 1 option (b): Score labels are the bare digits, read after an appended space id
    score = label_vocab(tok, "score", 10)
    assert score == tuple(tok.encode(str(d), add_special_tokens=False)[0] for d in range(10))
    assert 242 not in score  # the space token itself is never a label


def test_label_not_single_token_raises():
    from fmt import LabelNotSingleToken, label_vocab

    with pytest.raises(LabelNotSingleToken, match="' A'"):
        label_vocab(FakeTok(split={" A"}), "choice", 3)
    assert len(label_vocab(FakeTok(), "choice", 26)) == 26


def test_permutations_identity_reversed_cyclic():
    from fmt import permutations_for

    assert permutations_for(4, 1) == [(0, 1, 2, 3)]
    assert permutations_for(4, 2) == [(0, 1, 2, 3), (3, 2, 1, 0)]
    # the cyclic shift perm[d] = (d + 1) % n is NOT its own inverse, unlike the reversal (R6)
    assert permutations_for(4, 3)[2] == (1, 2, 3, 0)
    for p in permutations_for(5, 4):
        assert sorted(p) == list(range(5))
    assert len(set(permutations_for(5, 4))) == 4


def test_permutations_too_many_raises():
    from fmt import permutations_for

    assert permutations_for(2, 2) == [(0, 1), (1, 0)]
    with pytest.raises(ValueError):
        permutations_for(2, 3)  # for n = 2 the reversal and the shift by 1 are the same perm


# ---- part 2: render ------------------------------------------------------------------------------
# skeleton/fmt_ref.py is the independent oracle: fmt.py must produce the same ids from its own text.


def test_prefix_identical_across_questions(tok):
    from fmt import Question, render

    s = "The refund was issued twice."
    q1 = Question("q1", "noul", "Was it refunded?", ("yes", "no"))
    q2 = Question("q2", "choice", "Who pays?", ("shop", "bank", "customer"))
    a = render(tok, s, [q1], n_perms=1, max_context=4096)
    b = render(tok, s, [q2, q1], n_perms=1, max_context=4096)
    assert a.prefix_ids == b.prefix_ids == tuple(tok.encode(prefix_text(s)))
    assert a.prefix_ids[0] == tok.bos_token_id  # BOS on the prefix only


@pytest.mark.skipif(not GOLD_EVAL.exists(), reason="run: uv run python -m skeleton.prep_gold")
def test_branch_ids_never_retokenized_join(tok):
    from fmt import Question, render

    rows = [json.loads(line) for line in GOLD_EVAL.open()][:100]
    joined_differs = 0
    for i, r in enumerate(rows):
        q = Question(str(i), r["type"], r["question"], tuple(r["options"]))
        out = render(tok, r["state"], [q], n_perms=1, max_context=8192)
        (b,) = out.branches
        split = tok.encode(prefix_text(r["state"])) + tok.encode(
            branch_text(r["type"], r["question"], r["options"]), add_special_tokens=False
        )
        assert out.prefix_ids + b.token_ids == tuple(split), r["id"]
        assert b.token_ids[0] != tok.bos_token_id  # no BOS inside a branch
        joined = tok.encode(prefix_text(r["state"]) + branch_text(r["type"], r["question"], r["options"]))
        joined_differs += tuple(joined) != out.prefix_ids + b.token_ids
    print(f"rows where re-tokenizing the joined text would differ: {joined_differs} of {len(rows)}")


def test_score_branch_appends_space_id(tok):
    from fmt import Question, render

    q = Question("sev", "score", "How severe?", ("low", "mid", "high"), (0, 1, 2))
    (b,) = render(tok, "S", [q], n_perms=3, max_context=4096).branches
    text_ids = tok.encode(
        branch_text("score", "How severe?", ("low", "mid", "high"), (0, 1, 2)), add_special_tokens=False
    )
    assert b.token_ids == tuple(text_ids) + (242,)  # MiniCPM5 space id, appended as an id (Amendment 1)
    assert b.label_ids == tuple(tok.encode(str(d), add_special_tokens=False)[0] for d in range(3))
    assert b.perm == (0, 1, 2) and b.qtype == "score"  # Score is never permuted


def test_render_perms_display_and_order(tok):
    from fmt import FORMAT_VERSION, Question, render

    c = Question("c", "choice", "Pick one", ("w", "x", "y", "z"))
    n = Question("n", "noul", "Sure?", ("yes", "no"))
    out = render(tok, "S", [c, n], n_perms=3, max_context=4096)
    assert out.format_version == FORMAT_VERSION and out.truncated is False
    assert [(b.question_id, b.perm) for b in out.branches] == [
        ("c", (0, 1, 2, 3)),
        ("c", (3, 2, 1, 0)),
        ("c", (1, 2, 3, 0)),
        ("n", (0, 1)),  # Noul: identity only, whatever n_perms is
    ]
    # display position d shows options[perm[d]]
    assert "A. x\nB. y\nC. z\nD. w\nAnswer:" in tok.decode(list(out.branches[2].token_ids))
    assert all(b.label_ids == out.branches[0].label_ids for b in out.branches[:3])  # letters follow display


def test_validation_errors(tok):
    from fmt import Question, TooManyOptions, render

    def r(q):
        return render(tok, "S", [q], n_perms=1, max_context=4096)

    with pytest.raises(TooManyOptions):
        r(Question("c", "choice", "Q", tuple(f"o{i}" for i in range(27))))
    with pytest.raises(TooManyOptions):
        r(Question("s", "score", "Q", tuple(f"o{i}" for i in range(11)), tuple(range(11))))
    with pytest.raises(ValueError):
        r(Question("s", "score", "Q", ("a", "b"), (1, 0)))  # values must ascend
    with pytest.raises(ValueError):
        r(Question("s", "score", "Q", ("a", "b"), (0,)))  # one value per option
    with pytest.raises(ValueError):
        r(Question("n", "noul", "Q", ("no", "yes")))  # Noul options are fixed (ADR-0003)


GOLDEN = pathlib.Path(__file__).resolve().parent / "fixtures" / "fmt_v1_golden.json"


def test_golden_v1_frozen(tok):
    """nanohunch-fmt-v1 is frozen (plan Phase 2 step 12): render must reproduce the stored ids exactly."""
    from fmt import FORMAT_VERSION, Question, render

    golden = json.loads(GOLDEN.read_text())
    assert FORMAT_VERSION == golden["format_version"] == "nanohunch-fmt-v1"
    assert {c["question"]["type"] for c in golden["cases"]} == {"noul", "choice", "score"}
    for c in golden["cases"]:
        q = c["question"]
        r = render(
            tok,
            c["state"],
            [Question(q["id"], q["type"], q["text"], tuple(q["options"]), tuple(q["values"]))],
            n_perms=c["n_perms"],
            max_context=8192,
        )
        assert list(r.prefix_ids) == c["prefix_ids"], q["id"]
        got = [
            {"perm": list(b.perm), "token_ids": list(b.token_ids), "label_ids": list(b.label_ids)} for b in r.branches
        ]
        assert got == c["branches"], q["id"]


def test_state_too_long_raises(tok):
    from fmt import Question, StateTooLong, render

    q = Question("n", "noul", "Q?", ("yes", "no"))
    with pytest.raises(StateTooLong, match="> 1024"):
        render(tok, "word " * 5000, [q], n_perms=1, max_context=1024)
