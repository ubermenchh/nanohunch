# Walking skeleton: B0 and stock LoRA (plan Phase 1)

A walking skeleton is the thinnest pipeline that runs end to end: data in, model read, (later)
model trained, number out. These are the **reference numbers** Phase 2 (`engine.py`) and Phase 5
(`train.py`) must reproduce; they are not published test results (throwaway salt).

## B0: MiniCPM5-2B-Base, zero-shot, restricted label readout (2026-09-25)

Reader: `skeleton/b0_reader.py` (one full sequence per question). 1,000 eval items, both orders.

| type | n | acc | ece15 | reversed flip |
|---|---|---|---|---|
| noul (BoolQ) | 500 | 0.670 | 0.118 | n/a |
| choice (ARC) | 500 | 0.790 | 0.076 | 0.202 |
| all | 1000 | 0.730 | 0.067 | n/a |

Breakdown:
- **BoolQ is nearly "always yes".** The model picks yes on 97% of items; gold is yes on 64%. So
  0.670 is barely above the always-yes baseline (0.64), and mean confidence is 0.73. This is the
  constant yes/no bias ADR-0003 anticipated (fixed yes/no order), and it is what fine-tuning and a
  per-type temperature must fix.
- **ARC:** Easy 0.848 (flip 14.4%), Challenge 0.732 (flip 26.0%). Flips concentrate where the model
  is unsure (mean confidence 0.67 on Challenge vs 0.75 on Easy).
- **No positional preference in aggregate:** top-1 display positions A..D = 131/127/114/128 vs gold
  131/126/131/112. The 20% flip rate is item-level instability, not an "always A" habit.
- Below-chance tell (R6): noul 0.670 >= 0.55, choice 0.790 >= 0.35: **pass**.
- Determinism: rerun of the first 50 rows, max abs prob diff **0.0**.
- Wall time about 2 min 20 s per 1,000 sequences (the plan estimated about 4 min).

## Stock LoRA reference: `mlx_lm.lora`, 950 gold items, 950 iters (1 epoch, batch 1), lr 5e-6, rank 16, scale 20 (2026-09-25)

This is the `lora` block in `reports/skeleton.json`, and the run Phase 5 must reproduce.
Adapter: `runs/skeleton/`. Training 3 min, peak memory 6.1 GB.

| type | n | B0 acc | LoRA acc | delta | B0 ece15 | LoRA ece15 | B0 flip | LoRA flip |
|---|---|---|---|---|---|---|---|---|
| noul | 500 | 0.670 | **0.806** | **+13.6** | 0.118 | **0.045** | n/a | n/a |
| choice | 500 | 0.790 | 0.802 | +1.2 | 0.076 | 0.058 | 0.202 | 0.192 |
| all | 1000 | 0.730 | **0.804** | **+7.4** | 0.067 | **0.049** | n/a | n/a |

- Paired counts: noul LoRA-right/B0-wrong **118** vs B0-right/LoRA-wrong 50; choice 13 vs 7.
- The "always yes" habit is gone: predicted-yes rate 97% -> 63% (gold 64%).
- No learned position bias: top-1 A..D = 140/114/125/121 (gold 131/126/131/112); flips unchanged.
- Adapter applied: max abs prob diff 0.33 on the first 5 items (R3).
- Gate: LoRA within 3 pts of B0 on both types: **pass**. Below-chance tell: pass.
- lr 1e-5 (same recipe otherwise): noul 0.802, choice 0.800, flip 0.198, ece15 all 0.061. The two
  are within 3 items of each other (noise). lr 5e-6 was kept for slightly lower ECE. **Caveat:** that
  choice was made on this eval split; acceptable because skeleton numbers are a dev reference and
  are never published as test results.

## First attempt (kept for the record): 1,900 iters (2 epochs), lr 2e-5, validation split bug

Block `lora_lr2e-5_2ep` in `reports/skeleton.json`; adapter `runs/skeleton_lr2e-5_2ep/`.

| type | n | B0 acc | LoRA acc | delta | B0 ece15 | LoRA ece15 | B0 flip | LoRA flip |
|---|---|---|---|---|---|---|---|---|
| noul | 500 | 0.670 | 0.706 | **+3.6** | 0.118 | 0.207 | n/a | n/a |
| choice | 500 | 0.790 | 0.758 | **-3.2** | 0.076 | 0.111 | 0.202 | **0.340** |
| all | 1000 | 0.730 | 0.732 | +0.2 | 0.067 | 0.158 | n/a | n/a |

Paired counts: noul LoRA-right/B0-wrong 22 vs B0-right/LoRA-wrong 4; choice 29 vs 45.
Iteration-1500 checkpoint (lowest validation loss region): choice 0.766, noul 0.644, flip 0.340.

**Sanity rule fires (plan Phase 1 gate): choice is 3.2 pts below B0 (limit 3).** The plan says to
treat this as a bug, not a result, until template, split_mismatch and learning rate are checked:
- Template: passthrough verified (`'<s>X B'`); **split_mismatch = 0 of 1,000**: clean.
- Adapter applied: max abs prob diff 0.26 on the first 5 items: yes.
- Loss mask: 2 trained tokens per step (label plus EOS): the prompt is masked.
- **LoRA learned a position bias B0 did not have:** top-1 display positions A..D went from
  131/127/114/128 to 156/173/98/73 (gold 131/126/131/112); flips rose from 20% to 34%; choice
  confidence rose from 0.71 to 0.87 (overconfident, ECE worse). Training labels were balanced
  (A 112, B 124, C 106, D 107).
- Validation loss bottomed at 0.348 (iter 1400) and rose to 0.559 at the end: overfitting in epoch 2.
- **Bug in my data split:** `to_lora_jsonl.py` took the last 50 rows as `valid`, and `gold_train.jsonl`
  is ordered by source, so `valid` is 50 ARC-Challenge items only (and those 50 were left out of
  training). Validation loss therefore tracked one slice, not the mix.
- **Resolved** by the bounded debug above: split fixed (shuffle before taking `valid`), 1 epoch,
  lr 1e-5 or 5e-6. Three things changed at once (lr, epochs, and 50 more ARC-Challenge rows in
  training), so this does not isolate a single cause; the likeliest is too many large steps on
  950 items (2x the steps at 4x the lr), which overfit into a position habit.

## Notes
- Salt `nanohunch-skeleton-v0` (throwaway, not the Phase 3 salt). BoolQ has no `title` column, so
  `group_key` = sha256 of the passage. Train and eval share 0 group keys.
- **ARC tests knowledge, not reading:** the state is only the exam question, with no passage.
- `to_lora_jsonl.py` shuffles `gold_train.jsonl` (seed 1) before the 950/50 split; `split_mismatch = 0`.
- Stock training: `mlx_lm.lora --iters 950 --learning-rate 5e-6 --batch-size 1 --num-layers 16
  --max-seq-length 2048 --grad-checkpoint --mask-prompt -c configs/skeleton_lora.yaml`.

## Interpretation (yours, 3 lines)
1. Fine-tuning fixed a bias: On BoolQ, B0 answered "yes" 97% of the time (gold: 64%). One epoch of LoRA brought that to 63% and moved the accuracy from 0.670 to 0.806 and ECE from 0.118 to 0.045 (118 items fixed vs 50 broken, p < 1e-6). The model could read the passages but its default answer was wrong.
2. ARC barely moved: +1.2 pts (13 vs 7 items, p ≈ 0.26). ARC here has no passage, so it tests stored knowledge, and 450 training items can't add knowledge to a 2B model. This matches pngwn's finding that knowledge scales with parameter count, not with the readout or light tuning.
3. Order sensitivity didn't change (flips 20% -> 19%), and the failed 2-epoch run showed that hard labels on a small set can easily learn a position habit instead (flips 34%). Stock LoRA won't deliver order robustness; that has to come from Phase 5's soft labels and a new option order every epoch, and it needs watching on the learning curve.
