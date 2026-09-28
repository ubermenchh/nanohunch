"""Phase 1 walking skeleton (plan Phase 1). Modules are imported inside each test so tests for
steps not yet built fail on their own, with a clear ModuleNotFoundError."""

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILT = ROOT / "data" / "skeleton"


# ---- step 1: prep_gold ------------------------------------------------------------------------


def test_split_deterministic():
    from skeleton.prep_gold import split_of

    got = {split_of("boolq", "Tie", "nanohunch-skeleton-v0") for _ in range(3)}
    assert len(got) == 1 and got <= {"train", "eval"}


@pytest.mark.skipif(not (BUILT / "gold_eval.jsonl").exists(), reason="run: uv run python -m skeleton.prep_gold")
def test_groups_disjoint():
    def keys(name):
        return {json.loads(line)["group_key"] for line in (BUILT / name).open()}

    assert keys("gold_train.jsonl").isdisjoint(keys("gold_eval.jsonl"))


# ---- step 4: tiny_eval (reference metrics; calibrate.py must match these in Phase 3) -----------


def test_ece_perfect():
    from skeleton.tiny_eval import ece15

    assert ece15([1.0, 1.0], [1, 1]) == 0.0


def test_ece_known():
    from skeleton.tiny_eval import ece15

    assert abs(ece15([0.9] * 10, [1] * 8 + [0] * 2) - 0.1) < 1e-9


def test_flip_rate():
    from skeleton.tiny_eval import flip_rate

    assert flip_rate([0, 1, 2], [0, 2, 2]) == 1 / 3


# ---- step 6: to_lora_jsonl (stock LoRA data; R6 remap) -----------------------------------------


def test_target_remap():
    import random
    import re

    from skeleton.to_lora_jsonl import make_example

    rng = random.Random(0)
    for k in range(200):
        n = rng.randrange(3, 6)
        opts = [f"opt{k}_{i}" for i in range(n)]
        row = {"type": "choice", "state": "S", "question": "Q?", "options": opts, "gold_index": rng.randrange(n)}
        ex = make_example(row, rng)
        letter = ex["completion"].strip()
        line = re.search(rf"^{letter}\. (.*)$", ex["prompt"], re.M).group(1)
        assert line == opts[row["gold_index"]]
        assert sorted(ex["perm"]) == list(range(n))
    noul = {"type": "noul", "state": "S", "question": "Q?", "options": ["yes", "no"], "gold_index": 1}
    ex = make_example(noul, rng)
    assert ex["completion"] == " no" and ex["perm"] == [0, 1]


# ---- step 3: b0_reader (yours) -----------------------------------------------------------------
# No weights needed: a fake tokenizer and a fake model that returns fixed logits.

LABEL_IDS = {" A": 10, " B": 11, " C": 12, " D": 13, " yes": 20, " no": 21}


class FakeTok:
    def __init__(self, two_id_label=None):
        self.two_id_label = two_id_label

    def encode(self, text, add_special_tokens=True):
        if text == self.two_id_label:
            return [99, LABEL_IDS[text]]
        if text in LABEL_IDS:
            return [LABEL_IDS[text]]
        ids = [5 + (ord(c) % 3) for c in text[:8]]  # arbitrary non-label ids
        return ([1] if add_special_tokens else []) + ids


class PrefersA:
    """Logits favour the label " A" (display position 0) at every position: a pure position bias."""

    def __call__(self, ids):
        import mlx.core as mx

        logits = mx.zeros((1, ids.shape[1], 32))
        return logits + mx.array([3.0 if v == LABEL_IDS[" A"] else 0.0 for v in range(32)])


ROW = {
    "id": "t:0",
    "type": "choice",
    "state": "S",
    "question": "Q?",
    "options": ["w", "x", "y", "z"],
    "gold_index": 0,
}


def test_reader_reverse_maps_to_canonical():
    from skeleton.b0_reader import read

    fwd = read(PrefersA(), FakeTok(), ROW)
    rev = read(PrefersA(), FakeTok(), ROW, reverse=True)
    n = len(ROW["options"])
    assert abs(sum(fwd) - 1) < 1e-6 and abs(sum(rev) - 1) < 1e-6
    assert max(range(n), key=fwd.__getitem__) == 0  # A shows canonical option 0
    assert max(range(n), key=rev.__getitem__) == n - 1  # reversed: A shows canonical option n-1


def test_reader_label_must_be_single_token():
    from skeleton.b0_reader import read

    with pytest.raises(AssertionError):
        read(PrefersA(), FakeTok(two_id_label=" A"), ROW)
