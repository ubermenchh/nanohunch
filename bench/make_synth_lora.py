"""P0-8: synthetic 2k-token LoRA rows for the Mac soak (plan Phase 0 step 12).

200 train + 5 valid rows of {"prompt": <decoded random ids>, "completion": " A"}, trimmed so the
templated length (passthrough template + completion) is 2,000 to 2,047 tokens.

usage: uv run python -m bench.make_synth_lora [MODEL_DIR]
"""

import json
import pathlib
import random
import sys

from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "p0_synth"


def templated_len(tok, text: str) -> int:
    msgs = [{"role": "user", "content": text}, {"role": "assistant", "content": " A"}]
    # transformers 5 returns a dict by default; ask for the plain id list
    return len(tok.apply_chat_template(msgs, tokenize=True, return_dict=False))


def row(tok, rng: random.Random) -> dict:
    ids = [rng.randrange(1000, 30000) for _ in range(2300)]
    lo, hi = 1, len(ids)
    while lo < hi:  # longest prefix whose templated length fits under 2,048
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if templated_len(tok, tok.decode(ids[:mid])) <= 2047 else (lo, mid - 1)
    text = tok.decode(ids[:lo])
    n = templated_len(tok, text)
    assert 2000 <= n <= 2047, n
    return {"prompt": text, "completion": " A"}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else "runs/models/minicpm5-2b-base-raw")
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, k in [("train", 200), ("valid", 5)]:
        rows = [row(tok, rng) for _ in range(k)]
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"wrote {k} rows to {(OUT / f'{name}.jsonl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
