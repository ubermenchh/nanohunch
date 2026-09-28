"""Adapters for evaluation-only public benchmarks and the project's SemIf anchor."""

import hashlib
import json
from pathlib import Path

import numpy as np

from fmt import Question


def norm_hash(text: str) -> str:
    """Hash lowercased, whitespace-collapsed text for overlap checks (R22)."""
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def _item(item_id, group, state, prompt, options, qtype, gold, consensus, meta, values=()):
    from evaluate import EvalItem

    state_id = norm_hash(state)
    question = Question(str(item_id), qtype, prompt, tuple(options), tuple(values))
    return EvalItem(state_id, str(group or item_id), state, question, gold, consensus, meta)


def _distribution(raw, n):
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (n,) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("consensus distribution must contain one finite nonnegative value per option")
    return values / values.sum()


def _gold_index(raw, options):
    if isinstance(raw, bool):
        return 0 if raw else 1
    if isinstance(raw, int):
        return raw if 0 <= raw < len(options) else None
    text = str(raw)
    for i, option in enumerate(options):
        if text == option or text == option.split(":", 1)[0]:
            return i
    return None


def convert_pngwn(row: dict, fields: dict[str, str]):
    """Map a configured pngwn row; unsupported decision types are intentionally skipped."""

    def get(name, default=None):
        key = fields.get(name)
        return row.get(key, default) if key else default

    kind = str(get("type", "")).lower()
    aliases = {
        "yes_no": "noul",
        "boolean": "noul",
        "bool": "noul",
        "choice": "choice",
        "score": "score",
        "noul": "noul",
    }
    qtype = aliases.get(kind)
    if qtype is None:
        return None
    raw_options = get("options", ["yes", "no"] if qtype == "noul" else [])
    if isinstance(raw_options, dict):
        raw_options = list(raw_options.items())
    option_text = [
        str(o.get("text", o.get("description", o.get("label", o.get("id", ""))))) if isinstance(o, dict) else str(o)
        for o in raw_options
    ]
    values = (
        [int(o.get("value", i)) if isinstance(o, dict) else i for i, o in enumerate(raw_options)]
        if qtype == "score"
        else ()
    )
    if qtype == "noul":
        order = [option_text.index(x) for x in ("yes", "no")] if set(option_text) == {"yes", "no"} else [0, 1]
        options = ["yes", "no"]
    else:
        order = list(range(len(option_text)))
        options = option_text
    raw_gold = get("gold")
    if qtype == "noul" and isinstance(raw_gold, bool):
        gold = 0 if raw_gold else 1
    else:
        gold0 = _gold_index(raw_gold, option_text)
        gold = order.index(gold0) if gold0 in order else None
    dist = _distribution(get("distribution"), len(option_text))
    if dist is not None:
        dist = dist[order]
    return _item(
        get("id", row.get("id", "pngwn:unknown")),
        get("group_key", row.get("group_key")),
        str(get("state", "")),
        str(get("question", "")),
        options,
        qtype,
        gold,
        dist,
        {"source": "pngwn", "family": str(get("family", "unknown"))},
        values,
    )


def convert_semif(row: dict):
    """Convert a SemIf authored144 record while retaining its option-id ordering."""
    raw_options = row["options"]
    options = [f"{o['id']}: {o.get('description', o['id'])}" for o in raw_options]
    qtype = (
        "noul" if len(options) == 2 and [o["id"] for o in raw_options] in (["yes", "no"], ["no", "yes"]) else "choice"
    )
    gold = int(row["label"])
    if not 0 <= gold < len(options):
        raise ValueError(f"SemIf label outside options for {row['id']}")
    if qtype == "noul" and raw_options[0]["id"] == "no":
        options.reverse()
        gold = 1 - gold
    consensus = _distribution(row.get("target_distribution"), len(options))
    if consensus is not None and qtype == "noul" and raw_options[0]["id"] == "no":
        consensus = consensus[::-1]
    return _item(
        row["id"],
        row.get("group_id"),
        row["state"],
        row["question"],
        options,
        qtype,
        gold,
        consensus,
        {"source": "semif", "family": row["family"], "split": row.get("split", "test")},
    )


def convert_jevbench(row: dict):
    """Convert one JevBench public item; unknown task types remain excluded."""
    state = row["state"]
    if not isinstance(state, str):
        # Preserve structured public states in stable JSON rather than dropping nested evidence.
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    question = row.get("question", {})
    kind = str(question.get("type", "")).lower()
    if kind not in {"choice", "noul"}:
        return None
    labels = list(row.get("labels", []))
    criteria = question.get("criteria", {})
    options = [f"{label}: {criteria[label]}" if label in criteria else label for label in labels]
    expected = row.get("expected")
    if kind == "noul":
        if set(labels) != {"yes", "no"}:
            return None
        order = [labels.index("yes"), labels.index("no")]
        options = ["yes", "no"]
        gold = order.index(labels.index(expected)) if expected in labels else None
    else:
        gold = labels.index(expected) if expected in labels else None
    return _item(
        row["id"],
        row.get("group"),
        state,
        question.get("instructions", ""),
        options,
        kind,
        gold,
        None,
        {"source": "jevbench", "family": row.get("family", "unknown"), "tier": row.get("split", "public")},
    )


def collect_eval_only_hashes(paths: list[Path]) -> list[str]:
    """Collect normalized state and question hashes from JSONL benchmark files."""
    hashes = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            state = row.get("state")
            question = row.get("question")
            if isinstance(question, dict):
                question = question.get("instructions", "")
            for text in (state, question):
                if isinstance(text, str) and text.strip():
                    hashes.add(norm_hash(text))
    return sorted(hashes)


def write_eval_only_hashes(paths: list[Path], output: Path) -> int:
    hashes = collect_eval_only_hashes(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{value}\n" for value in hashes))
    return len(hashes)
