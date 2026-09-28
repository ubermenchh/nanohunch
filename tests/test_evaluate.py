"""Phase 3 evaluator contract: score public EvalItems through the Predictor boundary."""

import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_run_eval_scores_a_toy_gold_item_and_writes_outputs(tmp_path):
    """A basic eval run must preserve gold accuracy and leave auditable item output."""
    from evaluate import EvalItem, run_eval
    from fmt import Question

    class FirstOptionPredictor:
        name = "toy"

        def predict(self, state, questions, *, n_perms=1):
            return [SimpleNamespace(probs=np.array([0.9, 0.1]), confidence=0.9)]

    items = [
        EvalItem(
            "state-1",
            "group-1",
            "the evidence",
            Question("q1", "choice", "Which?", ("yes", "no")),
            0,
            None,
            {"source": "toy", "length_bucket": "le512"},
        )
    ]

    result = run_eval(
        FirstOptionPredictor(),
        items,
        cfg={"bins": 15, "flip_suite": [], "length_edges": [512, 2048, 4096], "coverages": []},
        out_dir=tmp_path,
    )

    assert result["overall"]["acc_gold"] == 1.0
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "items.jsonl").is_file()


def test_load_items_maps_built_jsonl_rows_to_eval_items(tmp_path):
    """The CLI loader preserves canonical labels, row identity and source metadata."""
    from evaluate import EvalItem, load_items
    from fmt import Question

    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "arc:42",
                "state_id": "state-hash",
                "group_key": "arc-group",
                "state": "Evidence",
                "qtype": "choice",
                "question": "Which?",
                "options": ["north", "south"],
                "gold": 1,
                "meta": {"source": "arc", "length_bucket": "le512"},
            }
        )
        + "\n"
    )

    items = load_items(path)

    assert items == [
        EvalItem(
            "state-hash",
            "arc-group",
            "Evidence",
            Question("arc:42", "choice", "Which?", ("north", "south")),
            1,
            None,
            {"source": "arc", "length_bucket": "le512"},
        )
    ]


def test_load_items_allows_rows_without_optional_meta(tmp_path):
    """A public-gold row may keep source at top level and omit the meta object."""
    from evaluate import load_items

    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "boolq:7",
                "group_key": "g7",
                "state": "Evidence",
                "qtype": "noul",
                "question": "True?",
                "options": ["yes", "no"],
                "gold": 0,
                "source": "boolq",
            }
        )
        + "\n"
    )

    assert load_items(path)[0].meta["source"] == "boolq"


@pytest.mark.parametrize(
    ("n_tokens", "expected"),
    [(512, "le512"), (513, "512_2k"), (2048, "512_2k"), (2049, "2k_4k"), (4096, "2k_4k"), (4097, "gt4k")],
)
def test_length_bucket_edges_are_inclusive(n_tokens, expected):
    """Bucket edges are inclusive so every model receives identical length slices."""
    from evaluate import length_bucket

    assert length_bucket(n_tokens, (512, 2048, 4096)) == expected


def test_remap_top1_uses_display_to_canonical_direction():
    """Non-self-inverse permutations catch remaps that accidentally use the inverse."""
    from evaluate import remap_top1

    assert remap_top1(0, (1, 2, 0)) == 1
    assert remap_top1(0, (2, 1, 0)) == 2


def test_run_eval_stores_canonical_top1_for_each_permutation(tmp_path):
    """Stored flip predictions must map display positions back to canonical option ids."""
    from evaluate import EvalItem, run_eval
    from fmt import Question

    class FirstOptionPredictor:
        name = "toy"

        def predict(self, state, questions, *, n_perms=1):
            q = questions[0]
            return [SimpleNamespace(probs=np.r_[0.9, np.full(len(q.options) - 1, 0.1 / (len(q.options) - 1))])]

    items = [
        EvalItem(
            f"state-{i}",
            f"group-{i}",
            "state",
            Question(f"q-{i}", "choice", "Choose", ("A", "B", "C")),
            0,
            None,
            {
                "source": "toy",
                "length_bucket": "le512",
                "perm_seed_1": "1,2,0",
                "perm_seed_2": "2,0,1",
                "perm_seed_3": "1,0,2",
            },
        )
        for i in range(2)
    ]
    cfg = {
        "bins": 15,
        "flip_suite": ["reverse", "perm_seed_1", "perm_seed_2", "perm_seed_3"],
        "length_edges": [512, 2048, 4096],
        "coverages": [],
    }

    metrics = run_eval(FirstOptionPredictor(), items, cfg=cfg, out_dir=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "items.jsonl").read_text().splitlines()]

    assert metrics["overall"]["acc_gold"] == 1.0
    assert metrics["flip"]["reverse"] == 1.0
    assert sum(row["flip_top1"]["perm_seed_1"] == 1 for row in rows) == 2
    for row in rows:
        for key in ("perm_seed_1", "perm_seed_2", "perm_seed_3"):
            perm = tuple(
                map(int, next(item.meta[key] for item in items if item.question.id == row["question_id"]).split(","))
            )
            assert row["flip_top1"][key] == perm[0]


def test_family_balanced_accuracy_weights_families_equally():
    """Macro averaging prevents a large family from hiding poor accuracy in a small one."""
    from evaluate import EvalItem, family_balanced_accuracy
    from fmt import Question

    items = [
        EvalItem(
            f"s{i}", f"g{i}", "state", Question(f"q{i}", "noul", "True?", ("yes", "no")), 0, None, {"family": family}
        )
        for i, family in enumerate(["large", "large", "large", "small"])
    ]
    correct = np.array([True, True, False, False])

    assert family_balanced_accuracy(items, correct) == pytest.approx(1 / 3)


def test_run_eval_reports_ece_by_option_count_bucket(tmp_path):
    """Separate option-count ECE reveals whether confidence changes with answer-set size."""
    from evaluate import EvalItem, run_eval
    from fmt import Question

    class FirstOptionPredictor:
        name = "toy"

        def predict(self, state, questions, *, n_perms=1):
            n = len(questions[0].options)
            return [SimpleNamespace(probs=np.r_[0.8, np.full(n - 1, 0.2 / (n - 1))])]

    items = [
        EvalItem(f"s{n}", f"g{n}", "state", Question(f"q{n}", "choice", "Pick", tuple(map(str, range(n)))), 0, None, {})
        for n in (2, 3, 5, 6)
    ]

    metrics = run_eval(
        FirstOptionPredictor(),
        items,
        cfg={"bins": 15, "flip_suite": [], "length_edges": [512, 2048, 4096], "coverages": []},
        out_dir=tmp_path,
    )

    assert metrics["option_count_buckets"]["2"]["n_items"] == 1
    assert metrics["option_count_buckets"]["3-5"]["n_items"] == 2
    assert metrics["option_count_buckets"]["6+"]["n_items"] == 1
    assert metrics["option_count_buckets"]["3-5"]["ece_width"] == pytest.approx(0.2)


def test_run_eval_baseline_aligns_ids_and_bootstraps_whole_groups(tmp_path):
    """Baseline pairing uses item IDs and resamples whole groups, not correlated decisions."""
    from evaluate import EvalItem, run_eval
    from fmt import Question

    class FirstOptionPredictor:
        name = "toy"

        def predict(self, state, questions, *, n_perms=1):
            return [SimpleNamespace(probs=np.array([0.9, 0.1]))]

    items = [
        EvalItem(
            f"s{group}-{i}",
            f"group-{group}",
            "state",
            Question(f"q{group}-{i}", "noul", "True?", ("yes", "no")),
            group % 2,
            None,
            {},
        )
        for group in range(40)
        for i in range(5)
    ]
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline = [
        {"state_id": item.state_id, "question_id": item.question.id, "correct_gold": False, "group_key": item.group_key}
        for item in reversed(items)
    ]
    (baseline_dir / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in baseline))

    metrics = run_eval(
        FirstOptionPredictor(),
        items,
        cfg={"bins": 15, "flip_suite": [], "length_edges": [512, 2048, 4096], "coverages": [], "bootstrap": 3000},
        baseline="baseline",
        out_dir=tmp_path / "current",
    )
    result = metrics["baseline"]["acc_gold"]

    assert result["delta"] == pytest.approx(0.5)
    assert result["hi"] - result["lo"] > 0.2


def test_run_eval_rejects_baseline_missing_item(tmp_path):
    """A partial baseline must fail instead of silently comparing mismatched populations."""
    from evaluate import EvalItem, run_eval
    from fmt import Question

    class FirstOptionPredictor:
        name = "toy"

        def predict(self, state, questions, *, n_perms=1):
            return [SimpleNamespace(probs=np.array([0.9, 0.1]))]

    item = EvalItem("state", "group", "state", Question("question", "noul", "True?", ("yes", "no")), 0, None, {})
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "items.jsonl").write_text("")

    with pytest.raises(ValueError, match="baseline baseline missing 1 items"):
        run_eval(
            FirstOptionPredictor(),
            [item],
            cfg={"bins": 15, "flip_suite": [], "length_edges": [512, 2048, 4096], "coverages": []},
            baseline="baseline",
            out_dir=tmp_path / "current",
        )


def test_run_eval_reports_overall_and_per_qtype_nll_brier_ece(tmp_path):
    """Overall and per-qtype nll/brier/ece let the M1 report show calibration by type."""
    from calibrate import brier as calib_brier
    from calibrate import nll as calib_nll
    from evaluate import EvalItem, run_eval
    from fmt import Question

    class FixedPredictor:
        name = "toy"

        def predict(self, state, questions, *, n_perms=1):
            q = questions[0]
            probs = np.array([0.7, 0.3]) if q.type == "choice" else np.array([0.6, 0.4])
            return [SimpleNamespace(probs=probs)]

    items = [
        EvalItem("s1", "g1", "state", Question("q1", "choice", "Which?", ("a", "b")), 0, None, {}),
        EvalItem("s2", "g2", "state", Question("q2", "choice", "Which?", ("a", "b")), 1, None, {}),
        EvalItem("s3", "g3", "state", Question("q3", "noul", "True?", ("yes", "no")), 0, None, {}),
    ]

    metrics = run_eval(
        FixedPredictor(),
        items,
        cfg={"bins": 15, "flip_suite": [], "length_edges": [512, 2048, 4096], "coverages": []},
        out_dir=tmp_path,
    )

    choice_probs = [np.array([0.7, 0.3]), np.array([0.7, 0.3])]
    assert metrics["by_qtype"]["choice"]["n_items"] == 2
    assert metrics["by_qtype"]["choice"]["nll"] == pytest.approx(calib_nll(choice_probs, [0, 1]))
    assert metrics["by_qtype"]["choice"]["brier"] == pytest.approx(calib_brier(choice_probs, [0, 1]))
    assert metrics["by_qtype"]["noul"]["n_items"] == 1
    all_probs = [np.array([0.7, 0.3]), np.array([0.7, 0.3]), np.array([0.6, 0.4])]
    assert metrics["overall"]["nll"] == pytest.approx(calib_nll(all_probs, [0, 1, 0]))
    assert metrics["overall"]["brier"] == pytest.approx(calib_brier(all_probs, [0, 1, 0]))
    assert metrics["overall"]["acc_gold"] == pytest.approx(2 / 3)


def test_run_eval_reports_accuracy_by_length_bucket(tmp_path):
    """Length-bucket accuracy lets the report show whether long inputs hurt correctness."""
    from evaluate import EvalItem, run_eval
    from fmt import Question

    class AlwaysFirst:
        name = "toy"

        def predict(self, state, questions, *, n_perms=1):
            return [SimpleNamespace(probs=np.array([0.9, 0.1]))]

    items = [
        EvalItem(
            "s1", "g1", "state", Question("q1", "noul", "True?", ("yes", "no")), 0, None, {"length_bucket": "le512"}
        ),
        EvalItem(
            "s2", "g2", "state", Question("q2", "noul", "True?", ("yes", "no")), 1, None, {"length_bucket": "le512"}
        ),
        EvalItem(
            "s3", "g3", "state", Question("q3", "noul", "True?", ("yes", "no")), 0, None, {"length_bucket": "gt4k"}
        ),
    ]

    metrics = run_eval(
        AlwaysFirst(),
        items,
        cfg={"bins": 15, "flip_suite": [], "length_edges": [512, 2048, 4096], "coverages": []},
        out_dir=tmp_path,
    )

    assert metrics["by_length_bucket"]["le512"] == {"n_items": 2, "acc_gold": pytest.approx(0.5)}
    assert metrics["by_length_bucket"]["gt4k"] == {"n_items": 1, "acc_gold": pytest.approx(1.0)}


def test_run_eval_reports_risk_coverage_at_configured_levels(tmp_path):
    """Risk-coverage shows whether abstaining on low-confidence items would raise accuracy."""
    from evaluate import EvalItem, run_eval
    from fmt import Question

    probs = [np.array([0.95, 0.05]), np.array([0.85, 0.15]), np.array([0.6, 0.4]), np.array([0.55, 0.45])]
    golds = [0, 0, 1, 1]  # items 0-1 correct (high confidence), 2-3 wrong (low confidence)

    class ListPredictor:
        name = "toy"

        def __init__(self):
            self._i = 0

        def predict(self, state, questions, *, n_perms=1):
            p = probs[self._i]
            self._i += 1
            return [SimpleNamespace(probs=p)]

    items = [
        EvalItem(f"s{i}", f"g{i}", "state", Question(f"q{i}", "noul", "True?", ("yes", "no")), gold, None, {})
        for i, gold in enumerate(golds)
    ]

    metrics = run_eval(
        ListPredictor(),
        items,
        cfg={"bins": 15, "flip_suite": [], "length_edges": [512, 2048, 4096], "coverages": [0.5, 1.0]},
        out_dir=tmp_path,
    )

    assert metrics["risk_coverage"]["0.5"] == pytest.approx(1.0)
    assert metrics["risk_coverage"]["1.0"] == pytest.approx(0.5)


def test_audit_agreement_reports_accuracy_per_type(tmp_path):
    """Skipped human judgments are excluded from each type's agreement denominator."""
    import csv

    from evaluate import audit_agreement

    path = tmp_path / "audit.csv"
    rows = [
        {
            "item_id": "c1",
            "qtype": "choice",
            "teacher_top1": "a",
            "consensus_top1": "a",
            "human_option_id": "a",
            "notes": "",
        },
        {
            "item_id": "c2",
            "qtype": "choice",
            "teacher_top1": "a",
            "consensus_top1": "b",
            "human_option_id": "b",
            "notes": "",
        },
        {
            "item_id": "c3",
            "qtype": "choice",
            "teacher_top1": "c",
            "consensus_top1": "c",
            "human_option_id": "c",
            "notes": "",
        },
        {
            "item_id": "c4",
            "qtype": "choice",
            "teacher_top1": "d",
            "consensus_top1": "d",
            "human_option_id": "d",
            "notes": "",
        },
        {
            "item_id": "n1",
            "qtype": "noul",
            "teacher_top1": "yes",
            "consensus_top1": "yes",
            "human_option_id": "yes",
            "notes": "",
        },
        {
            "item_id": "n2",
            "qtype": "noul",
            "teacher_top1": "no",
            "consensus_top1": "yes",
            "human_option_id": "yes",
            "notes": "",
        },
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    assert audit_agreement(path) == {"choice": (0.75, 4), "noul": (1.0, 2)}
