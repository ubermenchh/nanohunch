"""P0-6: the 0.12 gap between batched and one-at-a-time full re-encode (plan Phase 0 step 8; R9).

Same inputs as bench/equiv.py (random state of S tokens, B random tails of 32), comparing
batched naive re-encode (B rows of state + tail in one forward) with each row alone.
Toggles, in the plan's order:
  --B 2,4,16      does it appear at B = 2?
  --cpu           CPU backend (if CPU matches isolated, a Metal kernel is at fault)
  --naive-sdpa    fp32 reference attention in place of mx.fast.scaled_dot_product_attention
  --same-rows     B identical rows (state + tail 0) vs row 0 alone
  --full-head     apply the LM head at every position, then take the last (what equiv.py does);
                  default applies it at the last position only (what engine.py does)

usage: uv run python -m bench.anomaly MODEL S DTYPE [--B 2,4,16] [--cpu] [--naive-sdpa] [--same-rows]
"""

import argparse
import glob
import os

import mlx.core as mx
import mlx_lm.models.qwen3 as qwen3_mod
from mlx.utils import tree_map
from mlx_lm import load

from bench.equiv_real import naive_sdpa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("S", type=int)
    ap.add_argument("dtype", choices=["fp32", "bf16"])
    ap.add_argument("--B", default="2,4,16")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--naive-sdpa", action="store_true")
    ap.add_argument("--same-rows", action="store_true")
    ap.add_argument("--full-head", action="store_true")
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    if a.naive_sdpa:
        qwen3_mod.scaled_dot_product_attention = naive_sdpa
    path = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--Qwen--{a.model}/snapshots/*"))[0]
    model, tok = load(path)
    if a.dtype == "fp32":
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
    letters = mx.array([tok.encode(x)[0] for x in [" A", " B", " C", " D"]])

    def probs(rows):
        if a.full_head:
            logits = model(rows)[:, -1, :]  # [B, L, V] first: ~10 GB fp32 at B=16, L=1056
        else:
            h = model.model(rows)[:, -1, :]
            head = model.model.embed_tokens.as_linear if model.args.tie_word_embeddings else model.lm_head
            logits = head(h)
        return mx.softmax(logits[:, letters].astype(mx.float32), axis=-1)

    for b in [int(x) for x in a.B.split(",")]:
        mx.random.seed(0)
        state = mx.random.randint(1000, 30000, (1, a.S))
        tails = mx.random.randint(1000, 30000, (b, 32))
        if a.same_rows:
            tails = mx.repeat(tails[:1], b, axis=0)
        rows = mx.concatenate([mx.repeat(state, b, axis=0), tails], axis=1)
        batched = probs(rows)
        alone = mx.concatenate([probs(rows[i : i + 1]) for i in range(b)])
        per_row = mx.max(mx.abs(batched - alone), axis=-1).tolist()
        worst = max(range(b), key=lambda i: per_row[i])
        print(
            f"device={mx.default_device()} naive_sdpa={a.naive_sdpa} same_rows={a.same_rows} full_head={a.full_head} {a.dtype} "
            f"S={a.S} B={b}: batched_vs_alone={max(per_row):.2e} worst_row={worst} "
            f"row0={per_row[0]:.2e} rows_over_1e-3={sum(d > 1e-3 for d in per_row)}/{b}"
        )


if __name__ == "__main__":
    main()
