"""P0-1: are the nanohunch-fmt-v1 answer labels single tokens? (plan Phase 0 step 5, ADR-0003 rule 4)

Checks Choice (" A".." Z") and Noul (" yes", " no") alone and after "Answer:", then every Score
scheme of ADR-0003 Amendment 1 so the choice rests on measurements:
  digits       " 0".." 9" as single ids (the rule as first written)
  letters      option (a): Score reuses " A".." J", so it passes iff Choice passes
  space_digit  option (b): "Answer: " ends in one space id, each bare digit is one id, and
               encode("Answer: " + d) == encode("Answer:") + [space_id, digit_id]

usage: uv run python -m bench.check_labels [--score-scheme digits|letters|space_digit] REPO [REPO ...]
Exit 0 iff Choice, Noul and the selected Score scheme pass in every tokenizer. The default is
space_digit, the scheme chosen in ADR-0003 Amendment 1 (2026-09-24).
"""

import argparse
import sys

from transformers import AutoTokenizer

CHOICE = [f" {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
NOUL = [" yes", " no"]
DIGITS = [str(d) for d in range(10)]


def single_after_answer(tok, labels: list[str]) -> list[str]:
    """Labels failing: one id alone, and the same id when appended to "Answer:" (no merge)."""
    ctx = tok.encode("Answer:", add_special_tokens=False)
    bad = []
    for s in labels:
        ids = tok.encode(s, add_special_tokens=False)
        joint = tok.encode("Answer:" + s, add_special_tokens=False)
        if len(ids) != 1 or joint != ctx + ids:
            bad.append(f"{s!r} alone={ids} after_answer={joint[len(ctx) :]}")
    return bad


def space_digit(tok) -> tuple[list[str], int | None]:
    ctx = tok.encode("Answer:", add_special_tokens=False)
    with_space = tok.encode("Answer: ", add_special_tokens=False)
    bad: list[str] = []
    if len(with_space) != len(ctx) + 1 or with_space[: len(ctx)] != ctx:
        return [f"'Answer: ' -> {with_space}, not 'Answer:' + one space id"], None
    space_id = with_space[-1]
    for d in DIGITS:
        ids = tok.encode(d, add_special_tokens=False)
        joint = tok.encode("Answer: " + d, add_special_tokens=False)
        if len(ids) != 1 or joint != ctx + [space_id] + ids:
            bad.append(f"{d!r} alone={ids} after_answer_space={joint[len(ctx) :]}")
    return bad, space_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--score-scheme", choices=["digits", "letters", "space_digit"], default="space_digit")
    a = ap.parse_args()
    failures = 0
    for repo in a.repos:
        # from the HF cache; works with tokenizer-only snapshots (Qwen3-4B weights are not needed here)
        tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
        adds_bos = tok.bos_token_id is not None and tok.encode("x")[0] == tok.bos_token_id
        choice_bad, noul_bad = single_after_answer(tok, CHOICE), single_after_answer(tok, NOUL)
        digits_bad = single_after_answer(tok, [f" {d}" for d in DIGITS])
        sd_bad, space_id = space_digit(tok)
        score_bad = {"digits": digits_bad, "letters": choice_bad[:10], "space_digit": sd_bad}
        print(f"{repo}: adds_bos={adds_bos} bos_id={tok.bos_token_id} space_id={space_id}")
        for name, bad in [("choice", choice_bad), ("noul", noul_bad)] + [
            (f"score:{k}", v) for k, v in score_bad.items()
        ]:
            print(f"  {name:18s} {'PASS' if not bad else f'FAIL ({len(bad)})'}")
            for line in bad[:3]:
                print(f"      {line}")
        failures += len(choice_bad) + len(noul_bad) + len(score_bad[a.score_scheme])
    print("ALL SINGLE-TOKEN" if failures == 0 else f"{failures} FAILURES (score scheme: {a.score_scheme})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
