# Research notes (orchestrator, 2026-09-23)

Ground truth gathered by the orchestrator from web sources before the fleet ran.
Every agent should treat this file as the shared factual base alongside
`system-map.md` and `requirements.md`. All numbers here are MEASURED only in the
sense of "published by the named source"; none were reproduced by us.

## 1. The user and the goal

- Goal: learning + portfolio project. Replicate the idea behind TypeSafe's Jev
  ("System One model") using open-source small models, understand every piece,
  publish a model, code and evals. Not a commercial product. No uptime SLO.
- Domain: general purpose (arbitrary typed questions over arbitrary text state),
  like Jev.
- Compute: Apple M5 MacBook, 24 GB unified memory, 10 cores, macOS 26.6
  (MEASURED via sysctl). Plus rented cloud GPUs by the hour. No owned NVIDIA GPU.
- Budget: under 200 USD TOTAL for teacher-LLM API calls plus GPU rental.
  This is the binding constraint.
- Time: solo, about 20+ hours per week.
- Tooling on the machine: python3 3.14.7, uv installed (MEASURED).
- The user has prior local projects in `/Users/umangkaushik/fun/` that indicate
  skills: `mini-vllm` (inference engine), `linear-attention`, `gemma-mlx`,
  `mlx-learn`, `metal_kernels`, `gpt2-inference-engine`, `rl-wordle`,
  `dist-train`, `gemm-optimization`. Recon should check what is reusable.
- New project root: `/Users/umangkaushik/fun/nanohunch` (empty git repo).

## 2. What Jev is (target behavior to replicate)

Source: https://typesafe.ai/blog/introducing-system-one-models-and-jev (2026-09-15),
https://docs.typesafe.ai/, https://evals.typesafe.ai/

- Input: a `state` (text or structured program state) plus a list of typed
  `questions`. Output: typed answers with probabilities. No text generation.
- Three primitives:
  - Choice: pick one option from a caller-supplied list (max cardinality 255).
    Returns choice, probability distribution, confidence.
  - Score: level on a scale (example UI shows levels 0..4). Returns score,
    distribution over levels, confidence.
  - Noul: yes/no. Returns P(yes) in 0..1.
- All questions are evaluated in parallel and in isolation against the same
  state in one request. Adding questions "barely changes response time".
- Claimed latency 70 to 500 ms end to end. Price 0.042 USD per million input
  tokens, outputs free.
- Training method named RLCD (Reinforcement Learning for Calibrated Decisions),
  details not published. Claims calibrated, consistent probabilities.
- Their eval method: 4 workflows (security incidents, agent trace observability,
  invoice processing, customer service). Each task decomposed into many atomic
  Noul/Choice/Score questions combined by code. Reference labels are the
  AVERAGE of GPT-6 Astra and Claude Fable 5.1 answering every question at high
  thinking. "Accuracy" = agreement with that consensus, not ground truth.
- Their published averages over the 4 workflows (per case):
  | model | accuracy | cost/case | time/case |
  |---|---|---|---|
  | sol (OpenAI) workflow | 74.1% | 0.0836 | 23.3 s |
  | opus 5 workflow | 73.1% | 0.1761 | 37.8 s |
  | terra workflow | 67.9% | 0.0304 | 10.1 s |
  | Jev workflow | 67.8% | 0.0004 | 0.4 s |
  | sonnet 5 workflow | 67.8% | 0.1174 | 78.1 s |
  | luna workflow | 66.8% | 0.0033 | 12.9 s |
  | DS v4 flash workflow | 64.4% | 0.0059 | 51.9 s |
  | haiku 4.5 workflow | 53.6% | 0.0195 | 12.5 s |
  Every model was better as a decomposed workflow than as a single prompt.
- Their open-source LLM adapter: https://github.com/typesafe-ai/system-one-adapter-python
  (constrains LLMs to output decisions compatible with their API). Potentially
  useful as a teacher-labeling harness; verify license before use.
- Weak points we identified: no published calibration evidence (reliability
  diagrams / ECE), consensus-of-LLMs as the reference, workflows written by
  their own team.

## 3. Candidate open base models (HF model cards, fetched 2026-09-23)

| model | released | params | license | arch | relevant published numbers |
|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B (+ -Base) | 2026-02 | 4B LM (5B with vision) | Apache-2.0 | hybrid: 8 x (3 x Gated DeltaNet + 1 x gated attention), 32 layers, hidden 2560, vocab 248,320, 262k ctx | MMLU-Pro 79.1, GPQA-D 76.2, SuperGPQA 52.9 (thinking mode) |
| Qwen/Qwen3.5-0.8B / 2B / 9B (+ -Base) | 2026-02 | 0.8 / 2 / 9B | Apache-2.0 | same family | 9B MMLU-Pro 82.5 |
| openbmb/MiniCPM5-2B (+ -Base, -Midtrain, -SFT, -MLX) | 2026-09 | 2.52B (1.98B non-embedding) | Apache-2.0 | plain LlamaForCausalLM, 42 layers, GQA 16Q/2KV, 131k ctx | MMLU-Pro 70.8, GPQA-D 70.2 (vs Qwen3.5-4B 78.0 / 77.1 in the same table). Also MiniCPM5-1B exists. |
| google/gemma-4-E2B / E4B | 2026-04 | ~2 / ~4B effective | Apache-2.0 | multimodal | weaker than both above in MiniCPM's comparison table |
| LiquidAI/LFM2.5-2.6B, LFM2.5-8B-A1B | 2026-07/08 | 2.6B / 8B total 1B active | LFM Open License v1.0 | hybrid conv + GQA | weaker knowledge |
| LiquidAI/LFM2.5-Encoder-350M / 230M | 2026-07 | 350M / 230M | LFM Open License v1.0 | bidirectional LFM2 encoder, 8k ctx | 17-task fine-tune avg 81.02 (ModernBERT-large 81.68). Fast on CPU. |
| Qwen/Qwen3-0.6B-Base | 2025 | 0.6B | Apache-2.0 | standard transformer | used as arm B in pngwn report |

No small Qwen3.6 / Qwen3.8 exists (those start at 27B). Benchmarks above are
self-reported with generation/thinking; a single-forward-pass decision model
mostly inherits knowledge and reading comprehension, not agentic scores.

MLX note: `mlx-lm` exists and supports LoRA fine-tuning on Apple Silicon.
Whether it supports Qwen3.5 hybrid (Gated DeltaNet) layers and sequence
classification heads is UNVERIFIED. MiniCPM5-2B ships an official MLX build.

## 4. Prior art: pngwn "open-jev" and its controlled report (most important)

Sources: https://huggingface.co/spaces/pngwn/open-jev ,
https://huggingface.co/datasets/pngwn/typed-decisions-causal-experiment/blob/main/REPORT.md
(published ~2026-09-16/17).

Setup: corpus `pngwn/typed-decisions-v2` (train 31,109 states / 45,932
decisions; cal 3,356 / 5,087; test 3,387 / 5,214). Mix of synthetic ticket
workflows (severity, review, escalate, team, workflow4 multi-decision), MMLU-Pro
multiple choice ("choice"), HotpotQA yes/no ("noul"). Ticket component is
CC-BY-NC-4.0. `team` field is a leaked 1:1 lookup.

Three arms:
- A: scalar cross-encoder. Qwen3.5-4B-Base + LoRA r16/a32 + scalar `score` head
  (`Qwen3_5TextForSequenceClassification`), one scalar per (state, question,
  option). Trained on a100x4, global batch 8, 11,484 steps, 2 epochs.
- B: cached causal typed scorer. Qwen3-0.6B-Base + LoRA r16/a32, LM head FROZEN,
  options labelled with letters A..Z, loss = cross-entropy restricted to the
  candidate letter rows of the LM head. 2 epochs, 5,742 steps, 68 min on one
  A100. Trick: compute only candidate rows of the LM head from hidden states
  (cut training memory about 6x; full 248k-vocab logits OOM'd at 20.7 GiB).
- C: masked diffusion 381M. Not relevant for us.

Results on 5,214 test decisions:
| | A scalar 4B | B causal 0.6B | C masked 381M |
|---|---|---|---|
| accuracy | 0.8069 | 0.7518 | 0.6715 |
| NLL raw -> T-scaled | 0.658 -> 0.518 | 0.696 -> 0.640 | 0.911 -> 0.832 |
| ECE raw -> T-scaled | 0.069 -> 0.026 | 0.064 -> 0.015 | 0.065 -> 0.049 |
| fitted temperature | 2.33 | 1.59 | 2.08 |

- On state-derivable synthetic decisions all arms within about 0.006. The gap
  is world knowledge: MMLU-Pro choice A 0.625, B 0.415, C 0.108; HotpotQA noul
  A 0.939, B 0.870. Knowledge scales with parameter count.
- Order sensitivity: arm B under reversed option order flips 37.5% of gold
  rankings, top-1 agreement 0.664. Arm A is order invariant by construction.
- Set-size sensitivity: arm A gold mass drops to 0.49 at 26 candidates vs arm B
  0.61. Each fails differently.
- Compute: KV-cache branching removes duplicated context 3.7x at k=4 questions,
  14.1x at k=16. One-token decision branch adds about 0 extra forward passes.
  Letter options up to 26 are a free gather from one logits row. Multi-token
  option spans cost about 2 to 4 ms per option at n=32/64 on A100.
  Wall clock at 0.6B was dominated by per-forward overhead (~65 ms flat prefill
  128..2048 tokens), so ms numbers are forward-count, not FLOPs.
- Branching on Qwen3.5 hybrid requires copying full-attention KV AND conv +
  recurrent state of Gated DeltaNet layers (`Cache.reorder_cache` with a zero
  index does both). Verified equal to full re-encode to ~1e-7 in fp32.
  Qwen3.5 DeltaNet/conv layers fell back to slow reference PyTorch kernels.
- Open-jev PR eval (3,219 decisions, 500 real Gradio PRs, 100 to 16k tokens):
  scorer 0.740 accuracy vs 0.508 majority, ECE 0.047. BUT the untouched base
  model read via letter logits was as good on judgement questions and better
  above 2,000 tokens, because the fine-tune was trained with sequences truncated
  at 384 tokens. Lesson: train at realistic input lengths.
- Dependent decisions: iterative masked refinement removed constraint
  violations (14 -> 0 of 609) but did not change accuracy. Independent causal
  scoring had 3/609 violations. For us: handle dependencies in caller code.

## 5. Other prior art seen (low trust)

- `harshatheg/Qwen-2.5-1B-RLCD` (HF, mlx): inference-only "parallel constrained
  decoding" on Qwen2.5-1.5B-Instruct-4bit, KV broadcast + logit slicing, claims
  5.6 to 7x speedup on M4 Max vs autoregressive JSON. No training despite the
  name; GitHub link is a placeholder. Useful only as an MLX inference sketch.

## 6. Teacher-label cost anchors

- From Jev evals (section 2) the cheapest capable API teachers per full
  workflow case: luna 0.0033 USD, DS v4 flash 0.0059 USD, terra 0.0304 USD.
  A "case" there contains many questions, so per-question cost is lower.
  Per-token prices for these 2026 models were NOT fetched: ASSUMPTION.
- Open-weight teacher option: run e.g. Qwen3.8-27B or Qwen3.8-35B-A3B on a
  rented GPU and read full option distributions directly from logits (gives
  soft labels for free, no sampling). GPU hourly prices were NOT fetched:
  ASSUMPTION (order of 1 to 3 USD/hr for an A100/H100 class card on
  RunPod/Lambda/Modal/HF Jobs).
- Free public labelled data usable as hard labels: MMLU-Pro, HotpotQA, BoolQ,
  ANLI/MNLI, and similar. Check each license.
