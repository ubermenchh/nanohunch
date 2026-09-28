"""Local teacher spike (replaces P0-7; research note 2026-09-24-free-teacher-options.md).

Runs an open-weight instruct model in MLX on the 50 P0 items and reads each option's probability
from the full-vocabulary softmax at the first answer position (no top-k truncation):
  prompt  = chat template(system = qtype-aware instruction, user = nanohunch-fmt-v1 text)
  mass    = total probability on the option labels (single-id variants summed: "A", " A"; "yes", "Yes", ...)
  probs   = per-option mass, renormalized (the soft label)
Choice items are also read in reversed order and pooled log-linearly (plan Phase 4 two-order pooling).

Prints: mean candidate mass per qtype, accuracy vs gold (canonical, reversed, pooled), reversed-order
flip rate, run-to-run max diff (determinism), prompt tok/s, peak memory.

usage: uv run python -m bench.teacher_spike MODEL [--limit N] [--out reports/phase0/teacher_<name>.jsonl]
"""

import argparse
import json
import math
import pathlib
import time

import mlx.core as mx
from mlx_lm import load

from skeleton.fmt_ref import branch_text, prefix_text

ROOT = pathlib.Path(__file__).resolve().parent.parent
SYSTEM = {
    "choice": "Answer with only the letter of the correct option. No other text.",
    "noul": "Answer with only yes or no. No other text.",
}
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"  # Gemma 4's empty thought block
VARIANTS = {"noul": [["yes", "Yes", " yes", " Yes", "YES"], ["no", "No", " no", " No", "NO"]]}


def label_rows(tok, qtype: str, n: int) -> list[list[int]]:
    """Single-id token rows for each option; multi-id spellings are dropped, not guessed."""
    spellings = VARIANTS[qtype] if qtype == "noul" else [[chr(65 + i), f" {chr(65 + i)}"] for i in range(n)]
    rows = []
    for group in spellings:
        ids = sorted({e[0] for e in (tok.encode(s, add_special_tokens=False) for s in group) if len(e) == 1})
        assert ids, f"no single-id spelling in {group}"
        rows.append(ids)
    return rows


def teacher_read(model, tok, row: dict, order: list[int]) -> tuple[list[float], float, int]:
    """Probabilities in canonical option order, candidate mass, prompt length."""
    qtype, n = row["type"], len(row["options"])
    shown = [row["options"][i] for i in order]
    user = prefix_text(row["state"]) + branch_text(qtype, row["question"], shown)
    msgs = [{"role": "system", "content": SYSTEM[qtype]}, {"role": "user", "content": user}]
    # Thinking off, per family. Qwen3.x: enable_thinking=False renders an empty <think></think> block.
    # Gemma 4 ignores that flag, and its tokenize=True path omits the empty thought block; without it
    # the first predicted token is <|channel> (thinking) with probability 1.0, not the answer.
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    if "<|channel>" in text and not text.endswith(EMPTY_THOUGHT):
        text += EMPTY_THOUGHT
    ids = tok.encode(text, add_special_tokens=False)  # the template adds any BOS itself
    logits = model(mx.array(ids)[None])[0, -1].astype(mx.float32)
    p = mx.softmax(logits)
    per_display = [sum(p[i].item() for i in rs) for rs in label_rows(tok, qtype, n)]
    mass = sum(per_display)
    canon = [0.0] * n
    for disp, c in enumerate(order):  # display position -> canonical option
        canon[c] = per_display[disp] / mass if mass > 0 else 1 / n
    return canon, mass, len(ids)


def pool(a: list[float], b: list[float]) -> list[float]:
    logs = [(math.log(max(x, 1e-12)) + math.log(max(y, 1e-12))) / 2 for x, y in zip(a, b, strict=True)]
    z = [math.exp(v - max(logs)) for v in logs]
    return [v / sum(z) for v in z]


def argmax(v: list[float]) -> int:
    return max(range(len(v)), key=v.__getitem__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data" / "raw" / "p0_items.jsonl").open()]
    rows = rows[: a.limit] if a.limit else rows
    t0 = time.perf_counter()
    model, tok = load(a.model)
    print(f"loaded in {time.perf_counter() - t0:.1f}s; peak {mx.get_peak_memory() / 1e9:.1f} GB")

    out, n_tok, t_run, det = [], 0, 0.0, 0.0
    for i, r in enumerate(rows):
        n = len(r["options"])
        t = time.perf_counter()
        fwd, mass_f, L = teacher_read(model, tok, r, list(range(n)))
        t_run += time.perf_counter() - t
        n_tok += L
        rec = {"id": r["id"], "type": r["type"], "gold": r["gold_index"], "fwd": fwd, "mass_fwd": mass_f}
        if r["type"] == "choice":
            rev, mass_r, L = teacher_read(model, tok, r, list(range(n))[::-1])
            n_tok += L
            rec |= {"rev": rev, "mass_rev": mass_r, "pooled": pool(fwd, rev)}
        if i < 3:  # determinism: identical request twice
            again, _, _ = teacher_read(model, tok, r, list(range(n)))
            det = max(det, max(abs(x - y) for x, y in zip(fwd, again, strict=True)))
        out.append(rec)

    for qtype in ("noul", "choice"):
        rs = [o for o in out if o["type"] == qtype]
        if not rs:
            continue
        mass = sum(o["mass_fwd"] for o in rs) / len(rs)
        acc = sum(argmax(o["fwd"]) == o["gold"] for o in rs) / len(rs)
        line = f"{qtype:6s} n={len(rs)} mean_candidate_mass={mass:.3f} min_mass={min(o['mass_fwd'] for o in rs):.3f} acc_fwd={acc:.2f}"
        if qtype == "choice":
            acc_rev = sum(argmax(o["rev"]) == o["gold"] for o in rs) / len(rs)
            acc_pool = sum(argmax(o["pooled"]) == o["gold"] for o in rs) / len(rs)
            flip = sum(argmax(o["fwd"]) != argmax(o["rev"]) for o in rs) / len(rs)
            line += f" acc_rev={acc_rev:.2f} acc_pooled={acc_pool:.2f} flip_rev={flip:.2f}"
        print(line)
    print(
        f"determinism_max_diff={det:.2e} prompt_tok_s={n_tok / t_run:.0f} peak_mem={mx.get_peak_memory() / 1e9:.1f} GB"
    )
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text("".join(json.dumps(o) + "\n" for o in out))
        print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
