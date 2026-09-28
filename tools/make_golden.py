"""Freeze nanohunch-fmt-v1 (plan Phase 2 step 12): write tests/fixtures/fmt_v1_golden.json.

The first 20 gold eval rows (Choice rendered with n_perms=2) plus 4 hand-made Score questions, each
stored as the exact prefix_ids, token_ids and label_ids render() produces with the MiniCPM5 tokenizer.
test_golden_v1_frozen then fails on any change to the format. Regenerating this file means a new
format version (ADR-0003: one-way after Phase 5).

usage: uv run python -m tools.make_golden
"""

import json
import pathlib

from transformers import AutoTokenizer

from fmt import FORMAT_VERSION, Question, render

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "fmt_v1_golden.json"
SCORE = [
    ("How severe is the reported issue?", ("cosmetic", "minor", "degraded", "major", "outage"), (0, 1, 2, 3, 4)),
    ("How urgent is a reply?", ("no rush", "this week", "today"), (0, 1, 2)),
    ("How confident is the customer?", ("unsure", "fairly sure", "certain"), (0, 5, 9)),
    ("Rate the refund risk.", tuple(f"level {v}" for v in range(10)), tuple(range(10))),
]
SCORE_STATE = "Subject: charged twice\n\nHi, I was billed twice for my annual plan yesterday. Please fix this."


def case(tok, state: str, q: Question, n_perms: int) -> dict:
    r = render(tok, state, [q], n_perms=n_perms, max_context=8192)
    return {
        "state": state,
        "question": {"id": q.id, "type": q.type, "text": q.text, "options": list(q.options), "values": list(q.values)},
        "n_perms": n_perms,
        "prefix_ids": list(r.prefix_ids),
        "branches": [
            {"perm": list(b.perm), "token_ids": list(b.token_ids), "label_ids": list(b.label_ids)} for b in r.branches
        ],
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(ROOT / "runs" / "models" / "minicpm5-2b-base-raw"))
    rows = [json.loads(line) for line in (ROOT / "data" / "skeleton" / "gold_eval.jsonl").open()]
    rows = rows[:10] + [r for r in rows if r["type"] == "choice"][:10]  # both types (the file is sorted by source)
    cases = [case(tok, r["state"], Question(r["id"], r["type"], r["question"], tuple(r["options"])), 2) for r in rows]
    cases += [case(tok, SCORE_STATE, Question(f"score{i}", "score", t, o, v), 1) for i, (t, o, v) in enumerate(SCORE)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"format_version": FORMAT_VERSION, "tokenizer": "openbmb/MiniCPM5-2B-Base", "cases": cases}) + "\n"
    )
    print(f"wrote {len(cases)} cases ({sum(len(c['branches']) for c in cases)} branches) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
