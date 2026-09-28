"""Phase 4 data invariants: stable option ids, release safety, and gold sanity."""

import numpy as np
import pytest


def test_to_display_follows_display_to_canonical_permutation():
    """Targets must use perm[j], not the inverse permutation."""
    from dataset import to_display

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 9))
        target = np.asarray(rng.dirichlet(np.ones(n)))
        ids = tuple(f"q:{i}" for i in range(n))
        probs_by_id = dict(zip(ids, target, strict=True))
        perm = tuple(rng.permutation(n))

        result = to_display(probs_by_id, ids, perm)

        np.testing.assert_allclose(result, target[list(perm)], atol=1e-12)
        assert result.sum() == pytest.approx(1.0, abs=1e-12)


def test_build_row_assigns_option_ids_and_type_specific_canonical_orders():
    """Ingest makes IDs stable, shuffles Choice only, and sorts Score by value."""
    from dataset import build_row

    raw = {
        "state_id": "state-1",
        "group_key": "group-1",
        "split": "train",
        "source": "example/public",
        "source_revision": "rev-1",
        "source_license": "apache-2.0",
        "state": "Evidence text",
        "meta": {"template": "fixture", "workflow": "test", "length_bucket": "le512"},
        "decisions": [
            {
                "question_id": "q-choice",
                "qtype": "choice",
                "text": "Which?",
                "options": ["red", "blue", "green"],
                "gold_index": 1,
                "label_origin": "gold",
            },
            {
                "question_id": "q-noul",
                "qtype": "noul",
                "text": "True?",
                "options": ["yes", "no"],
                "gold_index": 0,
                "label_origin": "gold",
            },
            {
                "question_id": "q-score",
                "qtype": "score",
                "text": "How severe?",
                "options": ["minor", "low", "high"],
                "values": [2, 0, 4],
                "gold_index": 0,
                "label_origin": "spec",
            },
        ],
    }

    row = build_row(raw, salt="fixture-salt")
    repeated = build_row(raw, salt="fixture-salt")
    choice, noul, score = row["decisions"]

    assert [option["id"] for option in choice["options"]] == [
        "q-choice:0",
        "q-choice:1",
        "q-choice:2",
    ]
    assert set(choice["canonical_order"]) == {"q-choice:0", "q-choice:1", "q-choice:2"}
    assert choice["canonical_order"] == repeated["decisions"][0]["canonical_order"]
    assert choice["gold_option_id"] == "q-choice:1"
    assert noul["canonical_order"] == ["q-noul:0", "q-noul:1"]
    assert score["values"] == [0, 2, 4]
    assert score["canonical_order"] == ["q-score:1", "q-score:0", "q-score:2"]


def test_assert_row_releasable_rejects_noncommercial_license():
    """Non-commercial source rows must never enter the releasable training set."""
    from dataset import assert_row_releasable

    row = {
        "state_id": "row-nc-1",
        "source": "example/source",
        "source_license": "cc-by-nc-4.0",
        "split": "train",
        "decisions": [],
    }

    with pytest.raises(ValueError, match="row-nc-1.*source_license"):
        assert_row_releasable(row)


@pytest.mark.parametrize("source", ["pngwn/typed-decisions", "pngwn/typed-decisions-v2"])
def test_assert_row_releasable_rejects_denylisted_sources(source):
    from dataset import assert_row_releasable

    row = {
        "state_id": "row-denied",
        "source": source,
        "source_license": "apache-2.0",
        "split": "train",
        "decisions": [],
    }

    with pytest.raises(ValueError, match="row-denied.*source"):
        assert_row_releasable(row)


def test_assert_row_releasable_rejects_eval_only_training_source():
    from dataset import assert_row_releasable

    row = {
        "state_id": "row-eval-only",
        "source": "TIGER-Lab/MMLU-Pro",
        "source_license": "apache-2.0",
        "split": "train",
        "decisions": [],
    }

    with pytest.raises(ValueError, match="row-eval-only.*source"):
        assert_row_releasable(row)


def test_assert_row_releasable_rejects_unapproved_teacher():
    from dataset import assert_row_releasable

    row = {
        "state_id": "row-bad-teacher",
        "source": "nanohunch/synthetic-triage",
        "source_license": "apache-2.0",
        "split": "train",
        "decisions": [
            {
                "label_origin": "teacher",
                "teachers": [{"teacher_id": "gpt-x"}],
            }
        ],
    }

    with pytest.raises(ValueError, match="row-bad-teacher.*teacher_id"):
        assert_row_releasable(row)


def test_assert_row_releasable_accepts_allowlisted_public_row():
    from dataset import assert_row_releasable

    row = {
        "state_id": "row-allowed",
        "source": "example/public",
        "source_license": "apache-2.0",
        "split": "train",
        "decisions": [{"label_origin": "gold", "teachers": []}],
    }

    assert assert_row_releasable(row) is None


def test_below_chance_tell_detects_yes_no_index_inversion():
    """Swapping yes/no targets by index must look worse than chance on gold."""
    from calibrate import accuracy

    gold_ids = ["yes" if i % 2 == 0 else "no" for i in range(400)]
    gold_targets = [int(option_id == "no") for option_id in gold_ids]
    predictions = []
    for i, target in enumerate(gold_targets):
        predicted = 1 - target if i % 5 == 0 else target
        predictions.append(np.array([0.8, 0.2]) if predicted == 0 else np.array([0.2, 0.8]))
    swapped_targets = [1 - target for target in gold_targets]

    assert accuracy(predictions, gold_targets) == pytest.approx(0.8)
    assert accuracy(predictions, swapped_targets) == pytest.approx(0.2)

    rng = np.random.default_rng(1)
    choice_predictions = []
    stale_index_targets = []
    for _ in range(400):
        perm = rng.permutation(4)
        correct_display_index = int(np.flatnonzero(perm == 0)[0])
        probs = np.full(4, 0.2 / 3)
        probs[correct_display_index] = 0.8
        choice_predictions.append(probs)
        stale_index_targets.append(0)
    choice_accuracy = accuracy(choice_predictions, stale_index_targets)

    assert 0.18 <= choice_accuracy <= 0.32


def test_lookup_filter_drops_questions_determined_by_a_spec_field():
    from dataset import find_lookup_questions

    teams = {f"product-{i}": f"team-{i}" for i in range(6)}
    rows = []
    for i in range(40):
        product = f"product-{i % 6}"
        rows.append(
            {
                "state_id": f"state-{i}",
                "spec": {"product": product},
                "decisions": [
                    {
                        "label_origin": "teacher",
                        "question": "Which team owns this?",
                        "gold_option_id": teams[product],
                    },
                    {
                        "label_origin": "teacher",
                        "question": "Which queue handles this?",
                        "gold_option_id": "queue-b" if i == 0 else "queue-a",
                    },
                ],
            }
        )

    assert find_lookup_questions(rows) == {"Which team owns this?"}


def test_lookup_filter_requires_twenty_distinct_rows():
    from dataset import find_lookup_questions

    rows = [
        {
            "state_id": f"state-{i}",
            "spec": {"product": "product-a"},
            "decisions": [
                {
                    "label_origin": "teacher",
                    "question": "Which team owns this?",
                    "gold_option_id": "team-a",
                }
            ]
            * 2,
        }
        for i in range(10)
    ]

    assert find_lookup_questions(rows) == set()


def test_leaks_answer_verbatim_detects_gold_option_text_in_state():
    from dataset import leaks_answer_verbatim

    decision = {
        "qtype": "choice",
        "label_origin": "teacher",
        "gold_option_id": "q:1",
        "options": [
            {"id": "q:0", "text": "refund"},
            {"id": "q:1", "text": "replace the item"},
        ],
    }

    assert leaks_answer_verbatim("Customer asks us to replace the item today.", decision)
    assert not leaks_answer_verbatim("Customer asks for help with a damaged order.", decision)


def test_leaks_answer_verbatim_ignores_spec_gold_decisions():
    from dataset import leaks_answer_verbatim

    decision = {
        "qtype": "choice",
        "label_origin": "spec",
        "gold_option_id": "q:1",
        "options": [{"id": "q:1", "text": "refund eligible"}],
    }

    assert not leaks_answer_verbatim("Refund eligible under the policy.", decision)
