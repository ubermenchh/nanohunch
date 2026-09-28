# Phase 2: the hand-written inference core vs the skeleton (2026-09-25)

Core so far: `fmt.py` 91, `engine.py` 81, `calibrate.py` 41 lines (213 / 1,000). Tests: 33 pass.

## Correctness
- **Branch oracle (R8, as amended at P0-4):** on the CPU in fp32, `score` (state once, trim per
  question) vs `score_reencode` (full re-encode) on 8 questions over a 1k-token state: **4.5e-6**
  (limit 1e-3). Isolation (a question alone vs among 16): exactly **0.0**; repeat call identical.
- **Head path:** backbone + gathered label rows match the full model's logits (MiniCPM5 loads as
  `llama`: untied `lm_head`, no pre-head scale).

## Engine vs skeleton on 1,000 eval items (bf16, `cli.py eval`, 205 s)

| type | skeleton acc | engine acc | delta | skeleton flip | engine flip |
|---|---|---|---|---|---|
| noul | 0.670 | 0.672 | +0.2 pt | n/a | n/a |
| choice | 0.790 | 0.788 | -0.2 pt | 0.202 | 0.200 |
| all | 0.730 | 0.730 | 0.0 | n/a | n/a |

- Accuracy gate (|delta| <= 0.5 pt per type): **pass**. Flip gate (within 1 pt): **pass**.
- Per-item gate (max abs prob diff <= 2e-2): **fails on 27 of 1,000 items** (max 2.72e-2, p99
  2.3e-2, mean 7.3e-3). Top-1 changed on 9 items, all near 50/50.
- **Cause: bf16 arithmetic, not the engine.** The 5 worst items rescored in fp32 on the CPU: engine
  vs full re-encode **0.0 to 2.8e-6**. In bf16 the skeleton runs one full-length forward while the
  engine runs prefix and branch as two shorter forwards, so different kernel shapes round
  differently. P0-4 measured exactly this on real weights: **2.9e-2** in bf16. The plan's 2e-2 was
  set from the older synthetic-weight figure (1.3e-2), and ADR-0003 rule 2 already accepts this
  split: trainer and engine both use the split path, so it is part of the model, not an error.
- **Decision (author, 2026-09-25):** bf16 per-item tolerance is 3e-2 (just above the measured
  2.9e-2); the fp32 CPU oracle remains the correctness check. `bench/compare_skeleton.py`: **PASS**.

## Format frozen
`nanohunch-fmt-v1` is frozen by `tests/fixtures/fmt_v1_golden.json` (`tools/make_golden.py`): 24
cases, 34 branches (10 BoolQ, 10 ARC with `n_perms=2`, 4 hand-made Score questions including
non-contiguous values `(0, 5, 9)`). `test_golden_v1_frozen` fails on any change to the rendered ids.
The plan said "the first 20 gold rows"; those are all BoolQ (the file is sorted by source), so the
fixture takes 10 of each type. Regenerating the fixture means a new format version (ADR-0003).

## Latency (render + score, batch 1, bf16)

| workload | state tokens | questions | median ms | synthetic ms | ratio |
|---|---|---|---|---|---|
| W0 | 256 | 4 | 392 | 318 | 1.23x |
| W1 | 1024 | 16 | 1548 | 893 | **1.73x** |
| W2 | 8192 | 16 | 7266 | 6600 | 1.10x |

W1 is above the plan's 1.5x note threshold (not a stop). Split: prefill 622 ms for 1,051 tokens,
then 16 branches at 68 ms each (about 32 tokens per branch). The synthetic figure batched all 16
branches in one forward; the engine runs them one at a time (R9, batch 1), so per-branch overhead
(42 layers, one evaluation and one trim each) dominates. Prefill chunking is not the cause (one chunk
here). If W1 ever matters for a demo, batching the branches is the lever (P0-6 showed it is exact).
