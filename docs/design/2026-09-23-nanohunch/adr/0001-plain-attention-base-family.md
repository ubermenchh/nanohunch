# ADR 0001: Train and release a plain full-attention base (Qwen3 ladder), not the Qwen3.5 hybrid

**Status:** proposed (confirm after Phase 0 measurements M3, M4, M5 in `architecture.md` section 5)
**Date:** 2026-09-23
**Door:** one-way (expensive to reverse)
**Deciders:** @umang (author), sd-architect

## Context

The budget is 200 USD total, about 90 USD of it for rented GPUs (`REQ:296-303`).
A 4B LoRA run costs 1 to 8 A100-hours on normal kernels (DERIVED, `REQ:330-335`).
Qwen3.5 (the best published small model, MMLU-Pro 79.1, `RN:74`) is a hybrid
with 24 Gated DeltaNet layers and 8 attention layers (`SM:26-27`). Its training
path is slow on both backends: mlx-lm disables the Metal kernel in training and
runs a per-token Python loop (`mlx_lm/models/qwen3_5.py:193`, `SM:15-28`), and
transformers falls back to pure torch unless optional hub kernels load
(`SM:29-34`, assumption A3 unverified). A 3 to 10x slowdown turns a 4B run into
8 to 25 A100-hours (`REQ:451`), most of the GPU budget. The engine must also
branch the cached state per question; for DeltaNet layers that state is a
non-trimmable recurrent/conv cache (`mlx_lm/models/cache.py:146-147`,
`mlx_lm/models/qwen3_5.py:304-305`), a second mechanism on top of KV branching.

## Decision

We will fine-tune and release a plain full-attention decoder: Qwen3-4B-Base
(4.02B params, `Qwen3ForCausalLM`, Apache-2.0, MEASURED via HF API 2026-09-23)
by default, with Qwen3-0.6B-Base and Qwen3-1.7B-Base (same tokenizer and
architecture) for pilots. MiniCPM5-2B-Base (plain `LlamaForCausalLM`,
Apache-2.0, `SM:204`) replaces Qwen3-4B-Base if its Phase 0 B0 on the cal
split is within 1 point of Qwen3-4B-Base's.

## Alternatives considered

| Option | Why it lost |
|---|---|
| Qwen3.5-4B-Base | Slow training kernels on Mac (measured in code) and on CUDA unless unverified kernels load; 3 to 10x slowdown consumes 8 to 25 of about 36 to 75 affordable A100-hours (`REQ:325`); hybrid branching adds recurrent-state snapshots. Kept as a Phase 0 B0 reference because its MLX inference path uses the fast kernel |
| MiniCPM5-2B-Base as default | Fewer params (2.52B) where knowledge scales with params (`RN:121-123`); no Qwen3-4B comparison exists in our notes, so the default goes to the larger model until M4 measures both. About 1.8x faster on the Mac (DERIVED: 1,650 / 895 tok/s) and 3.4x smaller KV per token (DERIVED: 43,008 vs 147,456 B), so it wins any near-tie |
| Qwen3.5-0.8B / 2B-Base | Same kernel penalty as 4B, less knowledge |
| Gemma 4, LFM2.5 | Weaker in the MiniCPM comparison; LFM licence not reviewed (`RN:77-78`) |

## Consequences

**We gain:** training on standard attention kernels on both backends; KV-only
branching that can be implemented as "extend then trim" on mlx-lm's `KVCache`
(`mlx_lm/models/cache.py:375-381`); a 0.6B/1.7B/4B ladder with one tokenizer so
cheap pilots transfer to the release, and 0.6B reproduces pngwn arm B's base
(`RN:106`) as a pipeline sanity check.

**We accept:** likely lower knowledge than Qwen3.5-4B (published MMLU-Pro 79.1
for Qwen3.5-4B; no directly comparable Qwen3-4B number in our notes); an older
(2025-04) pretraining corpus; a larger KV cache than MiniCPM5 (144 KiB vs 42 KiB
per token, DERIVED) that forces micro-batching in the `batch` branch strategy at
long states.

**We are now locked into:** every paid training run, every adapter, and every
teacher soft label read in the student's tokenizer. Switching family after the
first full run costs a relabel of logit-read teacher labels only if the teacher
shares the tokenizer, plus 2 to 4 retrain runs (about 1 to 8 A100-hours each,
`REQ:335`).

**Revisit if:** Phase 0 M3 shows Qwen3.5-4B LoRA step time <= 1.5x Qwen3-4B at
seq 2k with hub kernels loaded AND M4 shows Qwen3.5-4B B0 >= best plain model
+ 3 points; or M5 shows Qwen3-4B W1 p50 > 2.5 s on the Mac (then MiniCPM5-2B).

## Verification

Phase 0 bake-off: B0 accuracy and T-scaled ECE on the cal split for
Qwen3-4B-Base, MiniCPM5-2B-Base and Qwen3.5-4B-Base with identical format and
readout; `mlx_lm.benchmark -p 1024` for W1 on the Mac; a 50-step LoRA timing
run on one rented GPU. The decision is failing if the first full Qwen3-4B run
gains < 1 point over its own B0 while Qwen3.5-4B's B0 alone is higher than
that run.

## Amendment 1 (2026-09-23, orchestrator, after red team; see `risks.md` R7)

The family decision (plain full attention, not Qwen3.5) stands. The default
member changes for the MVP:

- **MVP base: MiniCPM5-2B-Base**, trained in MLX on the M5 (MEASURED synthetic
  LoRA 232 to 347 tok/s at 4k to 1k, W1 893 ms, W2 6.6 s at 11.3 GB). Reason:
  it removes the rented-GPU trainer, the torch engine and MLX/torch parity from
  the MVP, which the red team showed is the difference between shipping and
  stalling.
- **Target-tier candidate: Qwen3-4B-Base**, trained on one Modal function, only
  if its Phase 0 B0 on the cal split beats MiniCPM5-2B-Base by more than 3
  points (reversed threshold from the original "within 1 point" rule: the
  cheaper model is now the default and the larger one must earn its cost).
- Qwen3-0.6B/1.7B pilots are dropped; pilots run on MiniCPM5-2B with 1k
  decisions (about 1.3 h on the Mac, DERIVED).
- Qwen3.5-4B stays only as an optional zero-train B0 reference row in the
  bake-off (inference path uses the fast Metal kernel).
