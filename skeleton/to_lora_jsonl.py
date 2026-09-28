"""Phase 1 step 6: gold train rows -> stock mlx_lm.lora completions data.

make_example(row, rng) -> {"prompt", "completion", "perm"}: Choice options are shown in a random
order, perm[display_pos] = canonical index (the registry's Perm convention), and the completion is
the label at the display position that holds the gold option. Noul is never permuted.

Writes data/skeleton/train.jsonl (first 950 train rows, what LoRA trains on) and valid.jsonl (last 50;
stock mlx_lm.lora requires one). Prints max templated length and split_mismatch: rows whose
chat-template ids differ from encode_split(prefix, branch) + [label_id] (ADR-0003 rule 2).

usage: uv run python -m skeleton.to_lora_jsonl [MODEL_DIR]
"""

import json
import pathlib
import random
import sys

from skeleton.fmt_ref import branch_text, label_strings, prefix_text

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "skeleton"


def make_example(row: dict, rng: random.Random) -> dict:
    n = len(row["options"])
    perm = rng.sample(range(n), n) if row["type"] == "choice" else list(range(n))
    shown = [row["options"][perm[j]] for j in range(n)]
    gold_display = perm.index(row["gold_index"])  # the j with perm[j] == gold_index
    prompt = prefix_text(row["state"]) + branch_text(row["type"], row["question"], shown)
    return {"prompt": prompt, "completion": label_strings(row["type"], n)[gold_display], "perm": perm}


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else "runs/models/minicpm5-2b-base-raw")
    rows = [json.loads(line) for line in (DATA / "gold_train.jsonl").open()]
    # gold_train.jsonl is ordered by source; shuffle so train and valid both mix BoolQ and ARC
    random.Random(1).shuffle(rows)
    rng = random.Random(0)
    exs = [make_example(r, rng) for r in rows]
    max_len, mismatch, dropped, kept = 0, 0, 0, []
    for r, ex in zip(rows, exs, strict=True):
        msgs = [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["completion"]}]
        templ = tok.apply_chat_template(msgs, tokenize=True, return_dict=False)
        shown = [r["options"][i] for i in ex["perm"]]
        split = (
            tok.encode(prefix_text(r["state"]))
            + tok.encode(branch_text(r["type"], r["question"], shown), add_special_tokens=False)
            + tok.encode(ex["completion"], add_special_tokens=False)
        )
        if len(templ) > 2048:
            dropped += 1
            continue
        max_len, mismatch = max(max_len, len(templ)), mismatch + (templ != split)
        kept.append({"prompt": ex["prompt"], "completion": ex["completion"]})
    (DATA / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in kept[:950]))
    (DATA / "valid.jsonl").write_text("".join(json.dumps(e) + "\n" for e in kept[950:1000]))
    print(f"max_len={max_len} split_mismatch={mismatch} of {len(kept)} dropped_over_2048={dropped}")
    print(f"wrote {len(kept[:950])} train, {len(kept[950:1000])} valid to {DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
