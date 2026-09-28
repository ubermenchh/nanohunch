"""25 BoolQ + 25 ARC-Challenge items for the Phase 0 probes (P0-4, P0-7) in the Phase 1 row schema.

Row: {"id", "group_key", "type": "noul" | "choice", "state", "question", "options", "gold_index"}.
Writes data/raw/p0_items.jsonl (gitignored).
"""

import hashlib
import json
import pathlib
import random

from datasets import load_dataset

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "p0_items.jsonl"
ARC_STATE = "Science exam question. There is no passage; answer from general knowledge.\n\n"


def boolq_row(i: int, r: dict) -> dict:
    q = r["question"].strip()
    title = r.get("title")
    return {
        "id": f"boolq:{i}",
        "group_key": title if title else "passage:" + hashlib.sha256(r["passage"].encode()).hexdigest()[:16],
        "type": "noul",
        "state": r["passage"],
        "question": q[0].upper() + q[1:] + ("" if q.endswith("?") else "?"),
        "options": ["yes", "no"],
        "gold_index": 0 if r["answer"] else 1,
    }


def arc_row(r: dict) -> dict:
    return {
        "id": f"arc-challenge:{r['id']}",
        "group_key": r["id"],
        "type": "choice",
        "state": ARC_STATE + r["question"],
        "question": "Which option correctly answers the question in the STATE?",
        "options": list(r["choices"]["text"]),
        "gold_index": r["choices"]["label"].index(r["answerKey"]),
    }


def main() -> None:
    boolq = load_dataset("google/boolq", revision="35b264d0", split="validation")
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", revision="210d026f", split="test")
    rng = random.Random(0)
    rows = [boolq_row(i, boolq[i]) for i in sorted(rng.sample(range(len(boolq)), 25))]
    arc_ok = [i for i in range(len(arc)) if 3 <= len(arc[i]["choices"]["text"]) <= 5]
    rows += [arc_row(arc[i]) for i in sorted(rng.sample(arc_ok, 25))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {OUT.relative_to(ROOT)}")
    for r in rows[:1] + rows[25:26]:
        print(f"  {r['id']}: {r['question']} -> {r['options'][r['gold_index']]!r}")


if __name__ == "__main__":
    main()
