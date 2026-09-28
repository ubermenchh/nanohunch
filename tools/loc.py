"""Core line budget (risks.md R23): non-blank, non-comment lines in the six core files.

Per-file budgets are soft (a warning); the 1,000-line total is hard (exit 1). When over budget:
delete first, then move glue out of the core, then raise a per-file target. Never the total.
"""

import pathlib
import sys

BUDGET = {"fmt.py": 130, "engine.py": 200, "calibrate.py": 170, "train.py": 220, "dataset.py": 150, "evaluate.py": 130}
TOTAL = 1000
ROOT = pathlib.Path(__file__).resolve().parent.parent


def loc(path: pathlib.Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#"))


def main() -> int:
    rows = {f: loc(ROOT / f) if (ROOT / f).exists() else 0 for f in BUDGET}
    for f, n in rows.items():
        print(f"{f:14s} {n:5d} / {BUDGET[f]:4d}{'  OVER (soft)' if n > BUDGET[f] else ''}")
    total = sum(rows.values())
    print(f"{'core total':14s} {total:5d} / {TOTAL}")
    return 1 if total > TOTAL else 0


if __name__ == "__main__":
    sys.exit(main())
