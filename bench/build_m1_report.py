"""Build M1 report figures and the derived tables run_eval does not compute directly
(reliability diagrams, JevBench per-tier ECE, SemIf/JevBench family-balanced accuracy).
Plan Phase 3 step 11 glue; run after the bake-off and anchor evals in reports/m1/.

usage: uv run python -m bench.build_m1_report
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
M1 = ROOT / "reports" / "m1"
BINS = 15


def load_items_jsonl(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "items.jsonl").read_text().splitlines() if line.strip()]


def reliability_diagram(run_dir: Path, bins: int = BINS) -> Path:
    """15 equal-width bins, confidence vs empirical accuracy, T-scaled confidence."""
    rows = [r for r in load_items_jsonl(run_dir) if r["correct_gold"] is not None]
    conf = np.array([max(r["probs"]) for r in rows])
    correct = np.array([r["correct_gold"] for r in rows], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    bin_acc = np.full(bins, np.nan)
    for b in range(bins):
        mask = idx == b
        if mask.any():
            bin_acc[b] = correct[mask].mean()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    ax.bar(
        edges[:-1], np.nan_to_num(bin_acc), width=1 / bins, align="edge", edgecolor="black", alpha=0.7, label="accuracy"
    )
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(run_dir.name)
    out = run_dir / "reliability.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def jevbench_tier_report(run_dir: Path, data_path: Path) -> dict:
    """Accuracy and ECE per source file (easy/original/hard); items.jsonl itself has no tier."""
    from calibrate import ece
    from evaluate import load_items

    items_by_id = {item.question.id: item for item in load_items(data_path)}
    rows = [r for r in load_items_jsonl(run_dir) if r["correct_gold"] is not None]
    by_tier: dict[str, list[dict]] = {}
    for row in rows:
        tier = items_by_id[row["question_id"]].meta.get("tier", "unknown")
        by_tier.setdefault(tier, []).append(row)
    report = {}
    for tier, trows in by_tier.items():
        conf = np.array([max(r["probs"]) for r in trows])
        correct = np.array([r["correct_gold"] for r in trows], dtype=float)
        report[tier] = {
            "n_items": len(trows),
            "accuracy": float(correct.mean()),
            "ece_width": ece(conf, correct, bins=BINS, scheme="width"),
        }
    return report


def family_balanced(run_dir: Path, data_path: Path) -> float:
    """SemIf's own metric: mean accuracy over task families, each family weighted equally."""
    from evaluate import family_balanced_accuracy, load_items

    rows_by_id = {r["question_id"]: r for r in load_items_jsonl(run_dir) if r["correct_gold"] is not None}
    items = [item for item in load_items(data_path) if item.question.id in rows_by_id]
    correct = np.array([rows_by_id[item.question.id]["correct_gold"] for item in items], dtype=float)
    return family_balanced_accuracy(items, correct)


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else "-" if x is None else str(x)


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_readme(summary: dict) -> None:
    """Assemble reports/m1/README.md from computed tables and figures (plan Phase 3 step 11).
    Leaves an Interpretation section for prose, which is not jarvis's to write (AGENTS.md Rule 2)."""
    import subprocess

    loc = subprocess.run(["uv", "run", "python", "tools/loc.py"], cwd=ROOT, capture_output=True, text=True).stdout
    latency_path = M1 / "latency.md"
    latency = latency_path.read_text() if latency_path.is_file() else "_latency.md not yet generated._"

    bakeoff_rows = []
    for name in ("minicpm5_b0", "qwen3_4b_b0", "qwen35_4b_b0"):
        m = summary.get(name)
        if not m:
            continue
        o = m["overall"]
        baseline = m.get("baseline") or {}
        delta = baseline.get("acc_gold", {})
        bakeoff_rows.append(
            [
                name,
                m["n_items"],
                o.get("acc_gold"),
                o.get("ece_width"),
                o.get("nll"),
                o.get("brier"),
                delta.get("delta"),
                delta.get("lo"),
                delta.get("hi"),
            ]
        )

    qtype_rows = []
    for name in ("minicpm5_b0", "qwen3_4b_b0", "qwen35_4b_b0"):
        m = summary.get(name)
        if not m:
            continue
        for qtype, v in m.get("by_qtype", {}).items():
            qtype_rows.append([name, qtype, v["n_items"], v["acc_gold"], v["ece_width"], v["nll"], v["brier"]])

    flip_rows = []
    for name in ("minicpm5_b0", "qwen3_4b_b0", "qwen35_4b_b0"):
        m = summary.get(name)
        if not m:
            continue
        for suite, v in m.get("flip", {}).items():
            flip_rows.append([name, suite, v])

    length_rows = []
    for name in ("minicpm5_b0", "qwen3_4b_b0", "qwen35_4b_b0"):
        m = summary.get(name)
        if not m:
            continue
        for bucket, v in m.get("by_length_bucket", {}).items():
            length_rows.append([name, bucket, v["n_items"], v["acc_gold"]])

    semif_rows = [
        ["MiniCPM5-2B-Base (ours, B0)", summary.get("minicpm5_semif_authored144", {}).get("family_balanced_accuracy")],
        ["MiniCPM5-2B (SemIf published, native BF16)", 0.686],
        [
            "Qwen3.5-4B-Base (ours, B0, re-encode)",
            summary.get("qwen35_4b_semif_authored144", {}).get("family_balanced_accuracy"),
        ],
        ["Qwen3.5-4B (SemIf published, native BF16)", 0.813],
        ["Qwen3-0.6B (SemIf published, native BF16, reference only)", 0.440],
    ]

    jev = summary.get("minicpm5_jevbench_public", {})
    jev_rows = [[tier, v["n_items"], v["accuracy"], v["ece_width"]] for tier, v in jev.get("by_tier", {}).items()]

    readme = f"""# M1: eval harness and B0 bake-off

Plan reference: `docs/design/2026-09-23-nanohunch/plan-parts/phase-3.md` step 11.

## Bake-off (test split, T-scaled)

{_table(["run", "n", "acc_gold", "ece_width", "nll", "brier", "baseline_delta", "ci_lo", "ci_hi"], bakeoff_rows)}

`baseline_delta` is each run's paired accuracy delta against `minicpm5_b0` (95% CI, group-bootstrapped
by state). Q4 (ADR-0001 Amendment 3): Qwen3-4B-Base was chosen as the MVP training base on the
**cal**-split comparison (delta +0.1855, CI +0.1581 to +0.2133); the test-split numbers above are
published but did not drive that decision.

## Accuracy and calibration by qtype

{_table(["run", "qtype", "n", "acc_gold", "ece_width", "nll", "brier"], qtype_rows)}

## Order-flip rates (Choice, >=3 options)

{_table(["run", "suite", "flip_rate"], flip_rows)}

## Accuracy by input length

{_table(["run", "length_bucket", "n", "acc_gold"], length_rows)}

## Reliability diagrams

![minicpm5_b0](minicpm5_b0/reliability.png) ![qwen3_4b_b0](qwen3_4b_b0/reliability.png) ![qwen35_4b_b0](qwen35_4b_b0/reliability.png)

## SemIf authored144 (mean family-balanced accuracy)

{_table(["run", "family_balanced_accuracy"], semif_rows)}

Checkpoint and prompt caveat: SemIf's published MiniCPM5-2B row uses `openbmb/MiniCPM5-2B`
(native BF16, chat-template JSON prompt); we only have `MiniCPM5-2B-Base` cached and score it
with the nanohunch prompt. Both checkpoint and serialization differ, so the gap below SemIf's
0.686 is not diagnosable as a readout bug from this comparison alone (bounded 3h audit done,
see `.jarvis/PROGRESS.md` Phase 3.7). The Qwen3.5-4B-Base row has the same caveat against SemIf's
0.813 (their row is the non-Base Qwen3.5-4B checkpoint).

## JevBench public items (self-run, not an official JevBench score)

{_table(["tier (source file)", "n", "accuracy", "ece_width"], jev_rows)}

The official JevBench score also covers sealed items, speed and cost; this is accuracy and
ECE on the public half only, self-run in our harness.

## pngwn

pngwn arm B is published at accuracy 0.752, ECE 0.015 (T-scaled), 37.5% order flips. We could not
run our own pngwn row for M1: `configs/eval_m1.yaml external.pngwn_test.fields` is still empty
because the pngwn schema (`data/raw/pngwn/typed-decisions-v2`) was unavailable during this phase.
The number above is a reference point, not a comparison on the same data.

{latency}

## Core line count

```
{loc.strip()}
```

## Contamination caveat

BoolQ, ARC, CommonsenseQA and HotpotQA (the four public sources behind `eval_v1`) are likely
present in both MiniCPM5-2B-Base's and Qwen3-4B-Base's/Qwen3.5-4B-Base's pretraining data. Accuracy
on these sources is not a clean held-out signal for either base model.

## Jev (footnote)

Jev's own published numbers are vendor-reported, not reproduced here.

## Calibration grouping (Phase 3.8, closed)

Checked `option_count_buckets` in the bake-off runs: bucket `2` is Noul-only, bucket `3-5` is
Choice-only, so the plan's within-qtype bucket-ECE comparison has nothing to compare on `eval_v1`.
Kept a single `T` per `(qtype, n_perms)`; re-check at Phase 6 step 15 once v2 adds 6-8 option
Choice items.

## Interpretation

_(not generated: report and model-card prose is written by hand, not by jarvis -- AGENTS.md Rule 2
and `.jarvis/PROJECT.md` Ownership. Add your read of the numbers above, then `git tag m1` and push.)_
"""
    (M1 / "README.md").write_text(readme)


def main() -> None:
    summary: dict = {}
    for name in ("minicpm5_b0", "qwen3_4b_b0", "qwen35_4b_b0"):
        run_dir = M1 / name
        if not (run_dir / "items.jsonl").is_file():
            continue
        reliability_diagram(run_dir)
        summary[name] = json.loads((run_dir / "metrics.json").read_text())

    semif = ROOT / "data" / "built" / "external" / "semif_authored144.jsonl"
    for name in ("minicpm5_semif_authored144", "qwen35_4b_semif_authored144"):
        run_dir = M1 / name
        if not (run_dir / "items.jsonl").is_file():
            continue
        reliability_diagram(run_dir)
        summary[name] = json.loads((run_dir / "metrics.json").read_text())
        summary[name]["family_balanced_accuracy"] = family_balanced(run_dir, semif)

    jevbench = ROOT / "data" / "built" / "external" / "jevbench_public.jsonl"
    run_dir = M1 / "minicpm5_jevbench_public"
    if (run_dir / "items.jsonl").is_file():
        summary["minicpm5_jevbench_public"] = json.loads((run_dir / "metrics.json").read_text())
        summary["minicpm5_jevbench_public"]["by_tier"] = jevbench_tier_report(run_dir, jevbench)

    (M1 / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_readme(summary)
    print(json.dumps({k: sorted(v) for k, v in summary.items() if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
