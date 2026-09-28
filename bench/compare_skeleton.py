"""Phase 2 gate: engine outputs vs the Phase 1 skeleton reference (plan Phase 2 step 10).

Per file (fwd, rev): rows matched by position and id; prints max abs prob diff and per-type accuracy
for both. Gate: per type |delta acc| <= 0.5 pt and max abs prob diff <= 3e-2 per item (bf16; raised
from 2e-2 on 2026-09-25 to the measured bf16 split gap, P0-4 2.9e-2).

usage: uv run python -m bench.compare_skeleton --ref reports/skeleton --new reports/phase2 [--name b0]
"""

import argparse
import json
import pathlib
import sys

import numpy as np


def acc(rows: list[dict], qtype: str) -> float:
    rs = [r for r in rows if r["type"] == qtype]
    return float(np.mean([int(np.argmax(r["probs"])) == r["gold_index"] for r in rs]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--name", default="b0")
    a = ap.parse_args()
    ok = True
    for side in ("fwd", "rev"):
        ref = json.load(open(pathlib.Path(a.ref) / f"{a.name}_{side}.json"))
        new = json.load(open(pathlib.Path(a.new) / f"{a.name}_{side}.json"))
        assert [r["id"] for r in ref] == [r["id"] for r in new], "row order differs"
        diffs = [
            max(abs(x - y) for x, y in zip(r["probs"], s["probs"], strict=True)) for r, s in zip(ref, new, strict=True)
        ]
        worst = int(np.argmax(diffs))
        print(f"{side}: max_abs_prob_diff={max(diffs):.2e} (row {ref[worst]['id']}) mean={np.mean(diffs):.2e}")
        for t in ("noul", "choice"):
            d = 100 * (acc(new, t) - acc(ref, t))
            print(f"  {t:6s} acc ref={acc(ref, t):.3f} new={acc(new, t):.3f} delta={d:+.2f} pt")
            ok &= abs(d) <= 0.5
        ok &= max(diffs) <= 3e-2
    print("PHASE 2 MATCH GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
