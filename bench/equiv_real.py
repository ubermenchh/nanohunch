"""P0-4: branch oracle on real MiniCPM5-2B weights (plan Phase 0 step 7; risks.md R8).

State = prefix_text of 5 BoolQ passages (about 1k tokens); 16 branches (8 BoolQ, 8 ARC), each read
through its own label rows at the last position. Measured in bf16 (for the record) and fp32:
  copy     prefill once, copy every layer's (k, v) into a fresh cache per branch (B = 1)
  trim     prefill once, run the branch on the shared cache, trim it back (what engine.py does)
  oracle   one full forward of prefix_ids + branch_ids, no cache
  isolation  branches cropped to the shortest length: batched 16 vs one at a time, on copied caches
Pass (R8): fp32 max abs prob diff (copy and trim vs oracle) <= 1e-3, and isolation == 0.0.

usage: uv run python -m bench.equiv_real [MODEL_DIR] [--naive-sdpa] [--cpu] [--branches N]
  --naive-sdpa  replace mx.fast.scaled_dot_product_attention with an fp32 reference (diagnostic)
  --cpu         run on the CPU backend (diagnostic: an independent kernel set)
"""

import argparse
import json
import pathlib
import sys

import mlx.core as mx
import mlx_lm.models.llama as llama_mod
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache, trim_prompt_cache

from skeleton.fmt_ref import branch_text, label_strings, prefix_text

ROOT = pathlib.Path(__file__).resolve().parent.parent


def naive_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """fp32 reference attention: GQA repeat, lower-right causal mask when the queries are the last L of S."""
    assert sinks is None
    dtype = queries.dtype
    q, k, v = (x.astype(mx.float32) for x in (queries, keys, values))
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
    scores = (q * scale) @ k.transpose(0, 1, 3, 2)
    L, S = q.shape[2], k.shape[2]
    if isinstance(mask, str) and mask == "causal":
        keep = mx.arange(S)[None, :] <= (mx.arange(L)[:, None] + S - L)
        scores = mx.where(keep, scores, -mx.inf)
    elif mask is not None:
        scores = mx.where(mask, scores, -mx.inf) if mask.dtype == mx.bool_ else scores + mask
    return (mx.softmax(scores, axis=-1) @ v).astype(dtype)


def label_ids(tok, qtype: str, n: int) -> list[int]:
    enc = [tok.encode(s, add_special_tokens=False) for s in label_strings(qtype, n)]
    assert all(len(e) == 1 for e in enc), enc
    return [e[0] for e in enc]


def readout(last_logits, ids):
    return mx.softmax(last_logits[..., mx.array(ids)].astype(mx.float32), axis=-1)


def copy_cache(cache, b: int = 1):
    out = []
    for c in cache:
        k, v = c.state
        y = KVCache()
        y.state = (mx.repeat(k, b, axis=0), mx.repeat(v, b, axis=0))
        out.append(y)
    return out


def measure(model, prefix, branches):
    base = make_prompt_cache(model)
    model(mx.array(prefix)[None], cache=base)
    mx.eval([c.state for c in base])
    p_len = base[0].offset
    assert p_len == len(prefix), (p_len, len(prefix))

    d_copy = d_trim = d_logit = 0.0
    top, distinct = [], set()
    trim_cache = make_prompt_cache(model)
    model(mx.array(prefix)[None], cache=trim_cache)
    for ids, labs in branches:
        full_last = model(mx.array(prefix + ids)[None])[0, -1]
        copy_last = model(mx.array(ids)[None], cache=copy_cache(base))[0, -1]
        oracle, copy = readout(full_last, labs), readout(copy_last, labs)
        trim = readout(model(mx.array(ids)[None], cache=trim_cache)[0, -1], labs)
        trim_prompt_cache(trim_cache, len(ids))
        assert trim_cache[0].offset == p_len
        d_copy = max(d_copy, mx.max(mx.abs(copy - oracle)).item())
        d_trim = max(d_trim, mx.max(mx.abs(trim - oracle)).item())
        d_logit = max(d_logit, mx.max(mx.abs(copy_last.astype(mx.float32) - full_last.astype(mx.float32))).item())
        top.append(mx.max(oracle).item())
        distinct.add(tuple(round(x, 4) for x in oracle.tolist()))
    # a degenerate readout (saturated or identical outputs) would make a 0.0 diff meaningless
    print(
        f"  {model.parameters()['model']['embed_tokens']['weight'].dtype}: full-vocab logit diff={d_logit:.2e} "
        f"top-prob mean={sum(top) / len(top):.3f} min={min(top):.3f} distinct_outputs={len(distinct)}/{len(branches)}"
    )

    width = min(len(ids) for ids, _ in branches)
    cropped = mx.array([ids[:width] for ids, _ in branches])
    lab = branches[0][1][:2]  # any fixed rows: isolation compares identical inputs, so rows do not matter
    batched = readout(model(cropped, cache=copy_cache(base, len(branches)))[:, -1], lab)
    alone = mx.concatenate(
        [readout(model(cropped[i : i + 1], cache=copy_cache(base))[:, -1], lab) for i in range(len(branches))]
    )
    d_iso = mx.max(mx.abs(batched - alone)).item()
    return d_copy, d_trim, d_iso


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?", default="runs/models/minicpm5-2b-base-raw")
    ap.add_argument("--naive-sdpa", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--branches", type=int, default=16)
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    if a.naive_sdpa:
        llama_mod.scaled_dot_product_attention = naive_sdpa
    print(f"device={mx.default_device()} naive_sdpa={a.naive_sdpa}")
    model_dir = a.model_dir
    rows = [json.loads(line) for line in (ROOT / "data" / "raw" / "p0_items.jsonl").open()]
    boolq, arc = [r for r in rows if r["type"] == "noul"], [r for r in rows if r["type"] == "choice"]
    model, tok = load(model_dir)
    k = 1  # passages until the prefix is about 1k real tokens (plan step 7)
    while len(tok.encode(prefix_text("\n\n".join(r["state"] for r in boolq[:k])))) < 1000 and k < len(boolq) - 8:
        k += 1
    prefix = tok.encode(prefix_text("\n\n".join(r["state"] for r in boolq[:k])))
    branches = []
    for r in boolq[-8:] + arc[:8]:
        ids = tok.encode(branch_text(r["type"], r["question"], r["options"]), add_special_tokens=False)
        branches.append((ids, label_ids(tok, r["type"], len(r["options"]))))
    branches = branches[: a.branches // 2] + branches[8 : 8 + a.branches - a.branches // 2]
    print(
        f"prefix_tokens={len(prefix)} branches={len(branches)} branch_len={min(len(b) for b, _ in branches)}..{max(len(b) for b, _ in branches)}"
    )

    c, t, i = measure(model, prefix, branches)
    print(f"bf16 copy_vs_reencode={c:.2e} trim_vs_reencode={t:.2e} isolation={i:.2e}")
    model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
    c, t, i = measure(model, prefix, branches)
    print(f"fp32 copy_vs_reencode={c:.2e} trim_vs_reencode={t:.2e} isolation={i:.2e}")
    ok = c <= 1e-3 and t <= 1e-3 and i == 0.0
    print("P0-4 PASS" if ok else "P0-4 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
