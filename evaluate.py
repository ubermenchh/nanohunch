import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from calibrate import brier, ece, nll, paired_bootstrap
from fmt import Question


@dataclass(frozen=True)
class EvalItem:
    """One canonical decision and its optional gold and consensus targets"""

    state_id: str
    group_key: str
    state: str
    question: Question
    gold: int | None
    consensus: np.ndarray | None
    meta: dict[str, str]


class Predictor(Protocol):
    """Common boundary for B0 and adapter predictors"""

    name: str

    def predict(self, state: str, question: Sequence[Question], *, n_perms: int = 1) -> Sequence[object]: ...


class B0Predictor:
    """Run base-model inference and expose T=1 pooled log-probabilities"""

    name = "b0"

    def __init__(self, model: str, calibration=None, *, reencode=False, max_context=8192, _adapter_path=None):
        # Keep base and adapter inference on the same scorer implementation
        from engine import MLXBranchScorer

        self.scorer = MLXBranchScorer(model, adapter_path=_adapter_path)
        self.calibration = calibration
        self.reencode = reencode
        self.max_context = max_context

    def predict_logprobs(self, state, questions, *, n_perms=1):
        """Return canonical pooled log-probabilities before temperature scaling"""
        from engine import pool
        from fmt import render

        # Render split-tokenized branches; the configured scorer handles cache limits
        rendered = render(self.scorer.tok, state, questions, n_perms=n_perms, max_context=self.max_context)
        score = self.scorer.score_reencode if self.reencode else self.scorer.score
        # Pool maps displayed permutation logits back to canonical option order
        return pool(score(rendered))

    def predict(self, state, questions, *, n_perms=1):
        """Return calibrated Answers while preserving the raw-logprob fitting path"""
        from calibrate import apply
        from engine import to_answer

        pooled = self.predict_logprobs(state, questions, n_perms=n_perms)
        answers = []
        for question in questions:
            z = pooled[question.id]
            if self.calibration is not None:
                probs = apply(self.calibration, question.type, n_perms, z)
            else:
                # Normalize in log space when no calibration artifact was requested
                probs = np.exp(z - z.max())
                probs /= probs.sum()
            answers.append(to_answer(question, probs))
        return answers


class AdapterPredictor(B0Predictor):
    """Run the shared predictor path with a trained adapter applied."""

    name = "adapter"

    def __init__(self, model, adapter, calibration=None, *, reencode=False, max_context=8192):
        # Forward the adapter into MLX model loading; otherwise this silently scores B0
        super().__init__(model, calibration, reencode=reencode, max_context=max_context, _adapter_path=adapter)


def load_items(path: str | Path) -> list[EvalItem]:
    """Load nanohunch eval JSONL rows into canonical EvalItems"""
    # Parse one persisted decision per nonblank line
    items = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qtype = row.get("qtype", row.get("type"))
        # Keep source metadata even when the builder stores it outside meta
        meta = dict(row.get("meta") or {})
        if "source" in row:
            meta.setdefault("source", row["source"])
        question = Question(
            str(row["id"]),
            qtype,
            row["question"],
            tuple(row["options"]),
            tuple(row.get("values", ())),
        )
        gold = row.get("gold", row.get("gold_index"))
        items.append(
            EvalItem(
                row.get("state_id", row["id"]),
                str(row.get("group_key", row["id"])),
                row["state"],
                question,
                gold,
                None if row.get("consensus") is None else np.asarray(row["consensus"]),
                meta,
            )
        )
    return items


def run_eval(
    pred: Predictor, items: Sequence[EvalItem], *, cfg: dict, baseline: str | None = None, out_dir: Path
) -> dict:
    """Score canonical predictions, returning gold accuracy and writing auditable JSON files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # Keep one paired flip result per eligible item and configured permutation.
    flip_results = {name: [] for name in cfg.get("flip_suite", [])}

    # Score the canonical order; permutation probes are added in the next evaluator step.
    for item in items:
        answer = pred.predict(item.state, [item.question], n_perms=1)[0]
        probs = np.asarray(answer.probs, dtype=float)
        correct = None if item.gold is None else int(np.argmax(probs)) == item.gold
        consensus_correct = None if item.consensus is None else int(np.argmax(probs)) == int(np.argmax(item.consensus))
        # Probe each displayed ordering, then map its top choice to canonical order.
        flip_top1 = {}
        q = item.question
        for name in flip_results:
            if q.type != "choice" or len(q.options) < 3:
                continue
            perm = (
                tuple(range(len(q.options) - 1, -1, -1))
                if name == "reverse"
                else tuple(map(int, item.meta[name].split(",")))
            )
            shown = Question(q.id, q.type, q.text, tuple(q.options[j] for j in perm), q.values)
            flipped = pred.predict(item.state, [shown], n_perms=1)[0]
            canonical = remap_top1(int(np.argmax(flipped.probs)), perm)
            flip_top1[name] = canonical
            flip_results[name].append(canonical != int(np.argmax(probs)))
        rows.append(
            {
                "state_id": item.state_id,
                "group_key": item.group_key,
                "question_id": item.question.id,
                "qtype": item.question.type,
                "source": item.meta.get("source"),
                "gold": item.gold,
                "probs": probs.tolist(),
                "correct_gold": correct,
                "correct_consensus": consensus_correct,
                "flip_top1": flip_top1,
            }
        )

    # Group gold-judged predictions by answer-set size; confidence is the top probability
    option_groups = {}
    for item, row in zip(items, rows, strict=True):
        n = len(item.question.options)
        bucket = "2" if n == 2 else "3-5" if 3 <= n <= 5 else "6+" if n >= 6 else None
        if bucket is None or row["correct_gold"] is None:
            continue
        option_groups.setdefault(bucket, []).append((max(row["probs"]), row["correct_gold"]))
    option_count_buckets = {}
    for bucket, values in option_groups.items():
        conf = np.array([value[0] for value in values])
        correct = np.array([value[1] for value in values])
        option_count_buckets[bucket] = {
            "n_items": len(values),
            "ece_width": ece(conf, correct, bins=cfg.get("bins", 15), scheme="width"),
            "ece_mass": ece(conf, correct, bins=cfg.get("bins", 15), scheme="mass"),
        }

    # Aggregate per qtype, length bucket and overrall so the M1 report can show calibration by slice
    overall_values, by_qtype_values, by_length_values = [], {}, {}
    for item, row in zip(items, rows, strict=True):
        if row["correct_gold"] is None:
            continue
        entry = (max(row["probs"]), row["correct_gold"], row["probs"], item.gold)
        overall_values.append(entry)
        by_qtype_values.setdefault(item.question.type, []).append(entry)
        bucket = item.meta.get("length_bucket")
        if bucket is not None:
            by_length_values.setdefault(bucket, []).append(entry)

    def summarize(values):
        conf = np.array([v[0] for v in values])
        correct = np.array([v[1] for v in values])
        return {
            "n_items": len(values),
            "acc_gold": float(correct.mean()),
            "ece_width": ece(conf, correct, bins=cfg.get("bins", 15), scheme="width"),
            "ece_mass": ece(conf, correct, bins=cfg.get("bins", 15), scheme="mass"),
            "nll": nll([v[2] for v in values], [v[3] for v in values]),
            "brier": brier([v[2] for v in values], [v[3] for v in values]),
        }

    overall_summary = summarize(overall_values) if overall_values else {"acc_gold": None}
    by_qtype = {qt: summarize(v) for qt, v in by_qtype_values.items()}
    by_length_bucket = {
        b: {"n_items": len(v), "acc_gold": float(np.mean([e[1] for e in v]))} for b, v in by_length_values.items()
    }
    risk_coverage = {}
    if overall_values:
        conf = np.array([v[0] for v in overall_values])
        correct = np.array([v[1] for v in overall_values])
        order = np.argsort(-conf, kind="stable")
        for coverage in cfg.get("coverages", []):
            k = max(1, round(coverage * len(order)))
            risk_coverage[str(coverage)] = float(correct[order[:k]].mean())

    # Align on stable item identity; row order can differ between runs
    baseline_metrics = None
    if baseline is not None:
        path = out_dir.parent / baseline / "items.jsonl"
        baseline_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        baseline_by_key = {}
        for row in baseline_rows:
            key = (row["state_id"], row["question_id"])
            if key in baseline_by_key:
                raise ValueError(f"baseline {baseline} duplicate item {key}")
            baseline_by_key[key] = row
        missing = [row for row in rows if (row["state_id"], row["question_id"]) not in baseline_by_key]
        if missing:
            raise ValueError(f"baseline {baseline} missing {len(missing)} items")
        pairs = [
            (row, baseline_by_key[(row["state_id"], row["question_id"])])
            for row in rows
            if row["correct_gold"] is not None
            and baseline_by_key[(row["state_id"], row["question_id"])]["correct_gold"] is not None
        ]
        if not pairs:
            raise ValueError(f"baseline {baseline} has no gold-judged aligned items")
        # Resample by state because decisions from one state are correlated
        delta, lo, hi = paired_bootstrap(
            np.asarray([row["correct_gold"] for row, _ in pairs], dtype=float),
            np.asarray([other["correct_gold"] for _, other in pairs], dtype=float),
            n=cfg.get("bootstrap", 10_000),
            seed=cfg.get("seed", 0),
            groups=np.asarray([row["group_key"] for row, _ in pairs]),
        )
        baseline_metrics = {"acc_gold": {"delta": delta, "lo": lo, "hi": hi}}

    metrics = {
        "predictor": pred.name,
        "n_items": len(rows),
        "overall": overall_summary,
        "by_qtype": by_qtype,
        "by_length_bucket": by_length_bucket,
        "risk_coverage": risk_coverage,
        "flip": {name: float(np.mean(values)) if values else None for name, values in flip_results.items()},
        "option_count_buckets": option_count_buckets,
        "baseline": baseline_metrics,
    }
    (out_dir / "items.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics


def length_bucket(n_tokens: int, edges: Sequence[int]) -> str:
    """Return the shared length slice; each configured edge belongs to the lower bucket."""
    # Reject invalid cuts so malformed configuration cannot silently change slices.
    if n_tokens < 0 or len(edges) != 3 or tuple(sorted(edges)) != tuple(edges):
        raise ValueError("expected a nonnegative length and three ascending edges")
    # Keep edges values inclusive so a token count at the boundary has one stable bucket.
    for label, edge in zip(("le512", "512_2k", "2k_4k"), edges, strict=True):
        if n_tokens <= edge:
            return label
    return "gt4k"


def remap_top1(display_top1: int, perm: Sequence[int]) -> int:
    """Map a displayed option index back to its canonical option index."""
    # 1. Check the permutation and index before lookup; negative indices would otherwise pass.
    if sorted(perm) != list(range(len(perm))) or not 0 <= display_top1 < len(perm):
        raise ValueError("invalid display index or option permutation")
    # 2. fmt.Perm stores canonical index by display position, so this lookup is direct.
    return perm[display_top1]


def family_balanced_accuracy(items: Sequence[EvalItem], correct: np.ndarray) -> float:
    """Average accuracy per family, giving each family equal weight."""
    values = np.asarray(correct, dtype=float)
    # Validate alignment so a family score cannot be paired with the wrong item.
    if values.shape != (len(items),):
        raise ValueError("correct must contain one value per item")
    by_family = {}
    # Collect correctness within each family before averaging; item counts may differ
    for item, value in zip(items, values, strict=True):
        family = item.meta.get("family")
        if not family:
            raise ValueError(f"missing family for {item.question.id}")
        by_family.setdefault(family, []).append(value)
    if not by_family:
        raise ValueError("at least one item is required")
    # Macro-average family scores so large families do not dominate the anchor metric
    return float(np.mean([np.mean(scores) for scores in by_family.values()]))
