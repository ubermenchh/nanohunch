import argparse
import json

import mlx.core as mx
from mlx_lm import load

from skeleton.fmt_ref import branch_text, label_strings, prefix_text


def encode_split(tok, prefix, branch):
    return tok.encode(prefix) + tok.encode(branch, add_special_tokens=False)


def read(model, tok, row, reverse=False):
    n = len(row["options"])
    order = list(range(n))[::-1] if reverse and row["type"] == "choice" else list(range(n))
    shown = [row["options"][i] for i in order]
    enc = [tok.encode(s, add_special_tokens=False) for s in label_strings(row["type"], n)]
    assert all(len(e) == 1 for e in enc), enc  # [0] of a 2-id label would read the space token
    label_ids = [e[0] for e in enc]
    ids = encode_split(tok, prefix_text(row["state"]), branch_text(row["type"], row["question"], shown))
    last = model(mx.array(ids)[None])[0, -1]  # logits at the last token of "Answer:"
    p = mx.softmax(last[mx.array(label_ids)].astype(mx.float32)).tolist()
    canon = [0.0] * n
    for disp, c in enumerate(order):  # display position -> canonical option
        canon[c] = p[disp]
    return canon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model, tok = load(a.model, adapter_path=a.adapter)
    rows = [json.loads(line) for line in open(a.data)]
    rows = rows[: a.limit] if a.limit else rows
    out = [
        {
            "id": r["id"],
            "type": r["type"],
            "gold_index": r["gold_index"],
            "reversed": a.reverse,
            "probs": read(model, tok, r, a.reverse),
        }
        for r in rows
    ]
    json.dump(out, open(a.out, "w"))
    print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
