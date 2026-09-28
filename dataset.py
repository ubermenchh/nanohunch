"""Data: splits, rows keyed by option id, teacher-label pooling"""

import hashlib
import random
from collections import defaultdict

import numpy as np

from fmt import Perm, QType

LICENSE_ALLOWLIST = frozenset({"cc-by-sa-3.0", "cc-by-sa-4.0", "mit", "apache-2.0"})
RELEASABLE_TEACHERS = frozenset({"deepseek-v4.1-flash", "qwen3.6-35b-a3b"})
DENY_SOURCES = frozenset({"pngwn/typed-decisions", "pngwn/typed-decisions-v2"})
EVAL_ONLY_SOURCES = frozenset({"TIGER-Lab/MMLU-Pro"})


def assert_row_releasable(row: dict) -> None:
    state_id = row.get("state_id", "<unknown>")
    source = row.get("source")
    if "source_license" not in row:
        raise ValueError(f"{state_id}: source_license")
    if source in DENY_SOURCES:
        raise ValueError(f"{state_id}: source")
    if row.get("split") == "train" and source in EVAL_ONLY_SOURCES:
        raise ValueError(f"{state_id}: source")
    if row["source_license"] not in LICENSE_ALLOWLIST:
        raise ValueError(f"{state_id}: source_license")
    for decision in row.get("decisions", []):
        teachers = decision.get("teachers", [])
        for teacher in teachers:
            if "teacher_id" not in teacher:
                raise ValueError(f"{state_id}: teacher_id")
            if decision.get("label_origin") == "teacher" and any(
                teacher["teacher_id"] not in RELEASABLE_TEACHERS for teacher in teachers
            ):
                raise ValueError(f"{state_id}: teacher_id")


def to_display(probs_by_option_id: dict[str, float], canonical_order: tuple[str, ...], perm: Perm) -> np.ndarray:
    if len(canonical_order) != len(perm) or set(perm) != set(range(len(canonical_order))):
        raise ValueError("perm must be a permutation of canonical option indices")
    if set(probs_by_option_id) != set(canonical_order):
        raise ValueError("probability option ids must match canonical_order")
    target = np.asarray([probs_by_option_id[option_id] for option_id in canonical_order])
    return target[np.asarray(perm)]


def find_lookup_questions(rows: list[dict]) -> set[str]:
    answers = defaultdict(lambda: defaultdict(set))
    rows_seen = defaultdict(set)

    for row in rows:
        for decision in row.get("decisions", []):
            if decision.get("label_origin") != "teacher":
                continue
            question = decision.get("question")
            answer = decision.get("gold_option_id")
            if question is None or answer is None:
                continue
            for field, value in row.get("spec", {}).items():
                key = (question, field)
                rows_seen[key].add(row["state_id"])
                answers[key][value].add(answer)
    return {
        question
        for (question, field), by_value in answers.items()
        if len(rows_seen[(question, field)]) >= 20
        and all(len(value_answers) == 1 for value_answers in by_value.values())
    }


def renormalize(top: list[dict], labels: list[str], qtype: QType) -> tuple[float, list[float]]:
    masses = np.zeros(len(labels), dtype=np.float64)
    for entry in top:
        token = entry["token"].strip().rstrip(".")
        for i, label in enumerate(labels):
            wanted = label.strip()
            matches = token.lower() == wanted.lower() if qtype == "noul" else token == wanted
            if matches:
                masses[i] += np.exp(entry["logprob"])
    total = float(masses.sum())
    probs = masses / total if total else np.full(len(labels), 1 / len(labels))
    return total, probs.tolist()


def pool_orders(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    if set(a) != set(b):
        raise ValueError("order distributions must have the same option ids")
    keys = list(a)
    with np.errstate(divide="ignore"):
        logits = (np.log([a[k] for k in keys]) + np.log([b[k] for k in keys])) / 2
    weights = np.exp(logits - np.max(logits))
    weights /= weights.sum()
    return dict(zip(keys, weights.tolist(), strict=True))


def consensus(per_teacher: list[dict[str, float]]) -> dict[str, float]:
    if not per_teacher or any(set(row) != set(per_teacher[0]) for row in per_teacher):
        raise ValueError("teacher distributions must share option ids")
    return {key: sum(row[key] for row in per_teacher) / len(per_teacher) for key in per_teacher[0]}


def leaks_answer_verbatim(state: str, decision: dict) -> bool:
    """Return whether a teacher-labelled answer appears verbatim in its evidence"""
    if decision.get("label_origin") != "teacher":
        return False
    gold_id = decision.get("gold_option_id")
    if gold_id is None:
        return False
    gold_text = next(
        (option.get("text", "") for option in decision.get("options", []) if option.get("id") == gold_id),
        "",
    )
    return bool(gold_text and gold_text.casefold() in state.casefold())


def build_row(raw: dict, *, salt: str) -> dict:
    row = {**raw, "schema_version": "nanohunch.data.v1"}
    decisions = []
    for source in raw["decisions"]:
        decision = dict(source)
        qid, qtype = decision["question_id"], decision["qtype"]
        options = decision["options"]
        ids = [f"{qid}:{i}" for i in range(len(options))]
        decision["options"] = [{"id": option_id, "text": text} for option_id, text in zip(ids, options, strict=True)]
        if qtype == "choice":
            order = list(range(len(ids)))
            seed = hashlib.sha256(f"{salt}|{row['state_id']}|{qid}".encode()).digest()
            random.Random(seed).shuffle(order)
        elif qtype == "noul":
            order = [0, 1]
        elif qtype == "score":
            values = decision["values"]
            order = sorted(range(len(ids)), key=values.__getitem__)
            decision["values"] = [values[i] for i in order]
        else:
            raise ValueError(f"{qid}: unknown qtype {qtype}")
        decision["canonical_order"] = [ids[i] for i in order]
        gold_index = decision.pop("gold_index", None)
        if gold_index is not None:
            decision["gold_option_id"] = ids[gold_index]
        decision.setdefault("gold_option_id", None)
        decision.setdefault("teachers", [])
        decision.setdefault("consensus_by_option_id", None)
        decisions.append(decision)
    row["decisions"] = decisions
    return row


def assign_split(group_key: str, source: str, fractions: dict[str, float], salt: str) -> str:
    s = sum(fractions.values())
    if abs(s - 1) > 1e-9:
        raise ValueError(f"split fractions sum to {s}, expected 1.0")
    h = hashlib.sha256(f"{salt}|{source}|{group_key}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    cum = 0.0
    for name, frac in fractions.items():
        cum += frac
        if u < cum:
            return name
    return name  # rounding left u past the end: the last split
