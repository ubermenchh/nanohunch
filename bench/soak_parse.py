"""P0-8: parse an mlx_lm.lora log into sustained-throughput numbers (plan Phase 0 step 12).

Wall time is accumulated from each report's It/sec over its span of iterations.
Pass rule (decision table): median tok/s after minute 30 >= 200, no OOM, drop <= 20%,
peak memory growth after minute 30 <= 0.5 GB.

Throughput is It/sec x tokens per iteration: with --mask-prompt, mlx-lm 0.31.3 reports only the
trained (unmasked) tokens in "Tokens/sec" (2 per step here), which says nothing about speed.

usage: uv run python -m bench.soak_parse LOG [TOKENS_PER_ITER]   (default: mean templated row length, 2,024)
"""

import re
import statistics
import sys

LINE = re.compile(r"Iter (\d+): Train loss [\d.naninf]+.*?It/sec ([\d.]+).*?Tokens/sec ([\d.]+).*?Peak mem ([\d.]+) GB")


def main() -> int:
    text = open(sys.argv[1]).read()
    per_iter = float(sys.argv[2]) if len(sys.argv) > 2 else 2024.0
    rows, prev_it, wall = [], 0, 0.0
    for m in LINE.finditer(text):
        it, its, mem = int(m[1]), float(m[2]), float(m[4])
        wall += (it - prev_it) / its
        prev_it = it
        rows.append((wall, its * per_iter, mem))
    if not rows:
        print("no Iter lines found")
        return 1
    late = [r for r in rows if r[0] >= 1800] or rows
    med = statistics.median(r[1] for r in late)
    mn = min(r[1] for r in late)
    drop = 100 * (1 - mn / med)
    mem30 = min(r[2] for r in late)
    mem_end = rows[-1][2]
    oom = "out of memory" in text.lower() or "insufficient memory" in text.lower()
    ok = med >= 200 and drop <= 20 and mem_end - mem30 <= 0.5 and not oom
    print(
        f"median_tok_s_after_30min={med:.0f} min_tok_s={mn:.0f} drop_pct={drop:.1f} peak_mem_30min={mem30:.2f} "
        f"peak_mem_end={mem_end:.2f} iters={prev_it} wall_min={wall / 60:.1f} oom={oom} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
