"""Diagnose SemIf parity by changing only Choice answer-token surface, not the frozen core."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np


def bare_choice_ids(tokenizer, count: int) -> tuple[int, ...]:
    """Resolve SemIf-style bare uppercase slots and reject non-single-token mappings."""
    ids = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{letter!r} is not one token: {encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != count:
        raise ValueError("bare answer-slot tokens collide")
    return tuple(ids)


def run_probe(model: str, data: Path, baseline_run: Path) -> dict:
    """Compare frozen spaced labels with bare labels on identical rendered prompts."""
    from engine import MLXBranchScorer, pool
    from evaluate import family_balanced_accuracy, load_items
    from fmt import render

    items = load_items(data)
    previous = {
        row["question_id"]: row for row in map(json.loads, (baseline_run / "items.jsonl").read_text().splitlines())
    }
    scorer = MLXBranchScorer(model)
    bare_correct = []
    for item in items:
        rendered = render(scorer.tok, item.state, [item.question], n_perms=1, max_context=8192)
        labels = bare_choice_ids(scorer.tok, len(item.question.options))
        branches = tuple(replace(branch, label_ids=labels) for branch in rendered.branches)
        bare_logprobs = pool(scorer.score(replace(rendered, branches=branches)))[item.question.id]
        bare_correct.append(int(np.argmax(bare_logprobs)) == item.gold)
    previous_correct = np.asarray([bool(previous[item.question.id]["correct_gold"]) for item in items])
    bare_correct = np.asarray(bare_correct)
    return {
        "model": model,
        "n_items": len(items),
        "existing_spaced_accuracy": float(previous_correct.mean()),
        "bare_label_accuracy": float(bare_correct.mean()),
        "existing_spaced_family_balanced_accuracy": family_balanced_accuracy(items, previous_correct),
        "bare_label_family_balanced_accuracy": family_balanced_accuracy(items, bare_correct),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openbmb/MiniCPM5-2B-Base")
    ap.add_argument("--data", type=Path, default=Path("data/built/external/semif_authored144.jsonl"))
    ap.add_argument("--baseline-run", type=Path, default=Path("reports/m1/minicpm5_semif_authored144"))
    ap.add_argument("--out", type=Path, default=Path("reports/m1/semif_label_probe.json"))
    args = ap.parse_args()
    result = run_probe(args.model, args.data, args.baseline_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
