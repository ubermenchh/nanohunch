# Phase 0: measurements and decisions

One row per measurement (plan Phase 0 step 13), filled in as each step runs.

**Phase 0 gate: confirmed by the author on 2026-09-25.** P0-1 Score scheme (b); P0-4 oracle on
CPU fp32 at 1e-3; P0-6 explained, batch 1 kept; P0-7' local teachers Gemma 4 + Qwen3.6
(ADR-0004 Amendment 2); P0-8 accepted as pass on intent (min 467 tok/s, memory flat).

| ID | Measurement | Value | Pass/fail | Decision taken |
|---|---|---|---|---|
| setup | mlx, mlx-lm, device | mlx 0.32.2, mlx-lm 0.31.3, `Device(gpu, 0)` | pass | none needed |
| setup | MiniCPM5-2B-Base tokenizer | adds BOS (`<s>`, id 0); space id 242 | info | `fmt.render` encodes the prefix with BOS (ADR-0003 rule 2 unchanged) |
| setup | Qwen3-4B-Base tokenizer | adds **no** BOS (`bos_token_id` is None); space id 220 | info | `bench/make_raw_model.py` uses `nobos` for Qwen3, `bos` for MiniCPM5 |
| P0-1 | Choice ` A`..` Z` and Noul ` yes`/` no`: one id each, alone and after `Answer:` | 28 of 28 single-token in both tokenizers | pass | none needed |
| P0-1 | Score ` 0`..` 9` (ADR-0003 rule 4 as written) | two ids each in both: MiniCPM5 `[242, digit]`, Qwen3 `[220, digit]` | **fail** | rule 4 amended: see next rows |
| P0-1 | Score option (a): letters ` A`..` J` | single-token in both | pass | candidate |
| P0-1 | Score option (b): `Answer: ` = `Answer:` + one space id, then bare digit `0`..`9` = one id, joint encode equal | holds for all 10 digits in both | pass | **chosen** 2026-09-24 (ADR-0003 Amendment 1); `check_labels` now defaults to it and exits 0 |

| setup | raw model dir | `runs/models/minicpm5-2b-base-raw`: passthrough template renders `'<s>X B'`, ids `[0, 77, 408]` == `encode('X B')` | pass | stock `mlx_lm` commands use this path |
| P0-4 | Branch oracle, real MiniCPM5 weights, 1,024-token prefix (5 BoolQ passages), 16 branches (18 to 58 tokens), GPU fp32 | copy and trim both 1.08e-3 vs re-encode (max abs prob); isolation 0.0; bf16 2.9e-2 | **fail** (limit 1e-3) | see cause and proposed decision below |
| P0-4 | Same, GPU fp32 with a naive fp32 reference attention in place of `mx.fast` SDPA | 1.35e-3 | fail | the fused attention kernel is not the cause |
| **CPU backend fp32**, 4 branches | full-vocab logit diff **0.0**, prob diff **0.0**; isolation 3.5e-6 | **pass** | branching is exact; the GPU gap is Metal fp32 accumulation order across matmul shapes. **Oracle runs on CPU fp32 from here on (R8 amended)** |

**P0-4 cause (2026-09-24).** Copy and trim agree with each other exactly, and on the CPU backend
branched and re-encoded fp32 logits are bit-identical, so the KV-cache branching math is correct.
On the Metal GPU, a 1,042-row forward and an 18-to-58-row branch run through differently tiled
fp32 kernels, which sum in a different order; 42 layers of real activations turn that into a
2e-2 full-vocab logit gap and 1.08e-3 on label probabilities. Not attention (the naive reference
SDPA gives 1.35e-3) and not RoPE (plain RoPE, theta 5e6, no scaling). A 477-token prefix gave
exactly 0.0 on the GPU, which is why length matters. Note: CPU bf16 isolation is 1.3e-2 while GPU
bf16 isolation is exactly 0, a second hint that kernel choice depends on shape.

**Decision (confirmed 2026-09-24):** keep R8's 1e-3 unchanged
and run the correctness oracle on the **CPU backend in fp32**, where branching is bit-exact. That
affects Phase 2 `test_oracle_fp32` and the Phase 5 `test_oracle_fp32_with_adapter`, about 8 items
each. The GPU fp32 figure stays in reports as information. Rejected: loosening to 2e-3 on the GPU
(the R8 rule forbids it, and it would stop detecting a real 1e-3 cache bug).

| P0-6 | Reproduce `bench/equiv.py Qwen3-0.6B 1024 16`: batched naive re-encode vs one row at a time | 1.20e-1 in fp32 and bf16 (matches the design) | reproduced | bisect |
| P0-6 | Same forward, LM head applied at the **last position only** (what `engine.py` and `train.py` do), GPU fp32, B = 2 to 16 | 9e-5 to 2.9e-4 (grows with B; same accumulation-order class as P0-4); CPU fp32 3e-6; same-rows B = 16 exactly 0 | pass | the backbone batches correctly |
| P0-6 | Head applied at **every position** first (`model(rows)[:, -1]`, as `equiv.py` does), GPU | fp32: 0.0 up to B = 12, **0.75** at B = 16; bf16: 0.0 up to B = 14, **0.12** at B = 16; CPU bf16 B = 16: 0.0 | **cause found** | see below |

**P0-6 cause (2026-09-24, 14 min of the 2 h box).** The 0.12 gap comes only from the
full-vocabulary logits tensor over all positions (`[16, 1056, 151936]`, about 2.6e9 elements) on
the Metal GPU. It appears abruptly between B = 14 and B = 16, while smaller batches are
bit-identical, and the CPU backend gives exactly 0.0. That points to a Metal kernel problem with
very large tensors rather than batching or attention (exact threshold and kernel not pinned down:
B = 15 untested). nanohunch never builds that tensor: ADR-0002 reads only the label rows at the
last position, and with that readout batched and single-row forwards agree to 2.9e-4 (fp32 GPU
noise). **Outcome A (explained).** R9 marked for re-review. Decision kept for the MVP: batch 1 plus
gradient accumulation (no cost, already planned) and the Phase 5 per-item NLL parity test.
Optional: a minimal upstream repro for `ml-explore/mlx` (the plan's rule: only if isolated to
Metal, which it is). Filing it is yours.

| P0-7' | Local teacher (replaces the OpenRouter probe: no account). `lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit` (Apache-2.0), full-vocab softmax at the answer position, 50 P0 items | Noul: candidate mass 0.997 (min 0.968), acc 0.88. Choice: mass 0.998 (min 0.987), acc 0.96 canonical / 0.80 reversed / **0.92 two-order pooled**, reversed flip 24%. Determinism diff 0.0. 439 prompt tok/s, peak 14.8 GB | **pass** (mass >= 0.9; repeatable by construction) | Gemma 4 is teacher 1. Two-order pooling for Choice stays (the teacher's own position bias is real). Qwen3.6-35B-A3B spike next for teacher 2 |
| P0-7' | Teacher 2: `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit` (Apache-2.0, 3-bit, `qwen3_5_moe`), same 50 items, thinking off via `enable_thinking=False` (2026-09-25) | Noul: mass 1.000, acc **0.72** (predicts yes 48%, gold 68%: leans No). Choice: mass 1.000, acc 1.00 canonical / 0.96 reversed / **1.00 pooled**, flip **4%**. 255 prompt tok/s, peak **16.8 GB** (about 2 GB under the 19.07 GB budget) | **pass** | Qwen3.6 is teacher 2 |
| P0-7' | Two-teacher consensus (mean of the two distributions; Choice uses each teacher's pooled orders) | Noul: top-1 agreement 0.76, consensus acc 0.84, both wrong 2/25. Choice: agreement 0.92, consensus acc 1.00, both wrong 0/25 | info | n = 25 per type: sanity only. The R11 human audit (Phase 4) decides whether consensus is a usable label |
| P0-7' | Repeatability | Both teachers bit-identical within a run and across separate processes on the same day (max diff 0.0). Gemma vs its first spike about 12 h earlier: max diff **0.025** with a byte-identical prompt and unchanged `uv.lock`; cause **unknown** | pass, with a caveat | Each teacher labels a dataset in one pass; the label cache records the run date so a drifted rerun is detectable |

**P0-7' notes (2026-09-24).** Gemma 4's chat template adds an empty thought block
(`<|channel>thought\n<channel|>`) only when rendered as text; `tokenize=True` drops it, and then the
first token is `<|channel>` with probability 1.0 (thinking), so the readout saw mass 0.000.
`bench/teacher_spike.py` renders text, appends the empty block if missing, and encodes. Noul leans
"No" mildly (predicts yes 64% vs gold 68%; one gold-yes item got 0.963 No), and Qwen3.6 leans "No"
more (48%): a constant yes/no bias that order pooling cannot remove. The two teachers disagree on
exactly those items, which is the point of having two. n = 25 per type, so these are sanity numbers, not estimates.
Volume estimate: about 14M prompt tokens for Phases 4 and 6 at about 440 tok/s is about 9 h of
unattended labelling per teacher (DERIVED); reusing the state's KV cache across its questions
would cut that, left for Phase 4.

| P0-8 | Stock `mlx_lm.lora` soak on real MiniCPM5 weights: 2,047-token rows, batch 1, 16 LoRA layers, rank 16, grad checkpoint, 1,950 iters, `caffeinate -i` (2026-09-25 01:03 to 02:45, 102 min) | prompt tok/s (It/sec x 2,047): minutes 0 to 60 about 470 to 560 (median about 520); minutes 60 to 102 about 710 to 890 (median about 860). Minimum after minute 30: **467**. Peak memory **8.52 GB flat** start to end. No OOM, no crash | **pass on intent** (see note): every report after minute 30 is >= 2.3x the 200 tok/s floor, memory growth 0.0 GB. The literal `drop <= 20%` rule reads FAIL (43%) only because the run got *faster* mid-way | Training stays on the Mac; the Modal kill criterion (A9/A10) does not fire. Phase 5 `cli.py train --tok-s` default becomes 500 (the conservative first-hour rate), replacing the synthetic 297 |

**P0-8 note.** `bench/soak_parse.py` measures the drop from the post-30-minute median; here that
median (824) is lifted by a speed-up at about minute 60 (about 520 to about 860 tok/s), so the
slowest early reports count as a "43% drop". The rule exists to catch throttling and allocator growth
over time; the trend here is upward and memory is flat. Cause of the first-hour slowness is
**unknown** (a guess: background load such as Spotlight indexing about 20 GB of fresh downloads).
mlx-lm's own `Tokens/sec` counts only unmasked (trained) tokens with `--mask-prompt` (0.85 here), so
throughput is computed from It/sec. Consequence: a 10k-decision epoch is about 5 to 8 h, not the
design's 14 h (DERIVED from 500 to 860 tok/s), so one epoch fits a night.

Commands: `uv run python -m bench.soak_parse runs/p0_soak/train.log 2047`;
`uv run python -m bench.teacher_spike MODEL [--limit N] [--out FILE]`;
`uv run python -m bench.anomaly Qwen3-0.6B 1024 fp32|bf16 [--B ...] [--cpu] [--naive-sdpa] [--same-rows] [--full-head]`;
`uv run python -m bench.equiv_real [--cpu] [--naive-sdpa] [--branches N]`;
`uv run python -m bench.check_labels [--score-scheme digits|letters|space_digit] openbmb/MiniCPM5-2B-Base Qwen/Qwen3-4B-Base`
(2026-09-24: `digits` 20 failures; `letters` and `space_digit` print `ALL SINGLE-TOKEN`, exit 0).
