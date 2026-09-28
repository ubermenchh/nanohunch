"""Phase 2 latency: render + score for W0/W1/W2 (plan Phase 2 step 11).

W0 = 256-token state, 4 questions; W1 = 1,024 and 16; W2 = 8,192 and 16. States are real P0 passages
repeated and cut to the exact token count. 1 warm-up, then the median of 5.

usage: uv run python -m bench.latency_p2 [--model M] [--out reports/phase2.md]
"""

import argparse
import json
import pathlib
import statistics
import time

import mlx.core as mx

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKLOADS = {"W0": (256, 4), "W1": (1024, 16), "W2": (8192, 16)}
SYNTH_MS = {"W0": 318, "W1": 893, "W2": 6600}  # capacity.md, synthetic weights


def main() -> None:
    from engine import MLXBranchScorer
    from fmt import Question, render

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/models/minicpm5-2b-base-raw")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--reencode", action="store_true", help="hybrid DeltaNet models cannot trim the KV cache (plan Phase 3 step 9)"
    )
    a = ap.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data" / "raw" / "p0_items.jsonl").open()]
    passages = "\n\n".join(r["state"] for r in rows if r["type"] == "noul")
    qrows = [r for r in rows if r["type"] == "noul"][:8] + [r for r in rows if r["type"] == "choice"][:8]
    sc = MLXBranchScorer(a.model)
    ids = sc.tok.encode(passages * 20, add_special_tokens=False)
    lines = ["| workload | state tokens | questions | median ms | synthetic ms | ratio |", "|---|---|---|---|---|---|"]
    for name, (n_tok, n_q) in WORKLOADS.items():
        state = sc.tok.decode(ids[:n_tok])
        qs = [Question(str(i), r["type"], r["question"], tuple(r["options"])) for i, r in enumerate(qrows[:n_q])]
        times = []
        for k in range(6):
            t = time.perf_counter()
            rendered = render(sc.tok, state, qs, n_perms=1, max_context=16384)
            sc.score_reencode(rendered) if a.reencode else sc.score(rendered)
            if k:
                times.append((time.perf_counter() - t) * 1e3)
        ms = statistics.median(times)
        lines.append(f"| {name} | {n_tok} | {n_q} | {ms:.0f} | {SYNTH_MS[name]} | {ms / SYNTH_MS[name]:.2f}x |")
        print(f"{name} {ms:.0f} ms (peak {mx.get_peak_memory() / 1e9:.1f} GB)", flush=True)
    table = "\n".join(lines)
    print(table)
    if a.out:
        kind = "re-encode" if a.reencode else "engine (branching)"
        with open(a.out, "a") as f:
            f.write(f"\n## Latency: {a.model} ({kind}, batch 1, bf16)\n\n" + table + "\n")


if __name__ == "__main__":
    main()
