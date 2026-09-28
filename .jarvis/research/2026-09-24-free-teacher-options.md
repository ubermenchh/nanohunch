# Free teacher-label options without OpenRouter (as of 2026-09-24)

**Question:** how to get soft teacher labels (label-row probabilities) from a releasable teacher
(ADR-0004) at zero cost, since the user has no OpenRouter credit. Also covers the synthetic-state
generator (plan Phase 4 step 5), which was planned on the DeepSeek API too.

**Hard requirement:** per-option probabilities, not just a text answer. That means top-k logprobs
from an API, or full logits from running the weights ourselves.

## What was checked (fetched 2026-09-24)

| Option | Verified facts | Fit |
|---|---|---|
| OpenRouter `:free` variants | 20 free variants in the public model list; only 5 advertise `logprobs`/`top_logprobs` (nex-agi n2.5 mini/pro, two inclusionai ling-3.0-flash variants, liquid lfm-2.5-2.6b). The one Qwen free variant (`qwen/qwen3.8-27b:free`) has no logprobs. Free variants have per-minute and per-day request caps that are lower if you have never bought credits (openrouter.ai/docs/api-reference/limits). | **No.** No releasable-licence teacher with logprobs; daily caps far below ~20k calls. |
| Hugging Face Inference Providers | Free accounts get **$0.10/month** of credits (huggingface.co/docs/inference-providers/pricing). | **No.** Pennies. |
| Cerebras | $5 free trial credit, needs a verified payment method, expires after 30 days; no permanent free tier; free-trial caps 1M tokens/day on `qwen-3.8-27b` and `gpt-oss-120b` (inference-docs.cerebras.ai/support/rate-limits). Logprob support not checked. | **Weak.** Not free, one-off, needs a card. |
| **Modal Starter plan** | **$30/month free compute**, 10 GPU concurrency; A100 80 GB $0.000694/s (about $2.50/h), H100 $0.001097/s (modal.com/pricing). The user already has a Modal account (`rl-wordle`). | **Yes.** vLLM on one A100 80 GB serves the full Qwen3.6-35B-A3B in FP8 (the weights are published) with logprobs. About 12 A100-hours/month free. |
| **Local MLX on the M5** (19.07 GB Metal working set) | `mlx-community/Qwen3.6-35B-A3B-4bit` is **20.4 GB: does not fit**. 3-bit builds fit: `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit` 17.4 GB, `andrevp/Qwen3.6-35B-A3B-3bit-MLX` 15.2 GB. `mlx-community/Qwen3-30B-A3B-4bit` 17.2 GB. Dense `Qwen3.6-27B` / `Qwen3.8-27B` MLX 4-bit about 16 GB. All Apache-2.0. Qwen3.6 models are `qwen3_5`/`qwen3_5_moe` (hybrid DeltaNet + attention, vocab 248,320). | **Yes, with caveats.** $0, deterministic, full label-row logits. 3-bit quantization costs some quality; speed on the M5 is unmeasured. |
| **Gemini API** (Google AI Studio) | Has a free tier (per-model caps shown only in AI Studio). **Supports logprobs:** `generationConfig.responseLogprobs` + `logprobs` top-k 0..20 (ai.google.dev/api/generate-content). Terms (effective 2026-03-23, ai.google.dev/gemini-api/terms): "You may not use the Services to develop models that compete with the Services" and "may not attempt to reverse engineer, extract or replicate any component of the Services, including the underlying data or models". Unpaid tier: Google uses prompts and outputs to improve its products; human reviewers may read them. | **Eval-only.** Technically fits, but the terms are not an explicit permission to distill into a released model (ADR-0004 requires one), and distillation reads close to "replicate". Fine for the B1 reference row. |
| **Gemma 4** (Google open weights) | **Apache-2.0**, not gated (HF API). `lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit` **15.6 GB** (MoE, about 4B active, quantization-aware trained, 380k downloads); `gemma-4-12B-it` MLX 4-bit 6.7 GB / 8-bit 12.7 GB; `gemma-4-31B-it` dense. | **Yes: best second teacher.** Fits the M5, releasable, and a different family from Qwen, which helps R11 (correlated teacher errors). |
| **Ollama** (local runner, llama.cpp/GGUF) | Current API returns `logprobs` with per-token `top_logprobs` on `/api/generate` and `/api/chat` (docs.ollama.com/api/generate, fetched 2026-09-24; the legacy `docs/api.md` does not list it). `raw: true` skips the chat template. Not installed on this Mac. Licensing = the model's own licence (open weights), so Gemma 4 / Qwen3.6 via Ollama are as releasable as via MLX. | **Usable, second choice.** Returns top-k of the *generated* token only, text in (its own tokenizer, no id-level control), an extra daemon plus a second (GGUF) copy of the weights. Direct MLX gives exact label-row probabilities for every option with the code we already have. Useful fallback if an MLX build is broken or a GGUF quant is better (for example unsloth's Qwen3.6 Q3/Q4). |
| Kaggle / Colab free GPUs | Not verified today (pages render client-side). T4-class 16 GB GPUs cannot hold a 35B MoE without heavy quantization. | **No** for this model size. |

## Takeaways
- No free API gives releasable-teacher logprobs at our volume. The free path is **running open
  weights ourselves**, which is what ADR-0004 originally decided ("bulk training labels with an
  open-weight teacher under Apache-2.0 or MIT, read through the nanohunch-fmt-v1 readout").
- Running the teacher ourselves is **better** than top-20 API logprobs: exact probabilities on the
  label rows (no top-20 truncation), fully deterministic (the P0-7 repeatability worry, R10,
  disappears), no provider pinning, and no Q6 terms question.
- Both paths also cover the synthetic-state generator (Qwen3.6-35B-A3B writes the tickets).
- Two teachers: two Qwen models are correlated (R11). **Gemma 4 26B-A4B (QAT 4-bit, 15.6 GB,
  Apache-2.0)** is the non-Qwen teacher that fits locally, so the pair Qwen3.6-35B-A3B (3-bit) +
  Gemma 4 26B-A4B covers two families at $0, run one after the other.

## Measured (2026-09-24/25, `reports/phase0.md` P0-7')
- Gemma 4 26B-A4B QAT 4-bit: 439 tok/s, 14.8 GB, noul 0.88, choice 0.92 pooled, flip 24%.
- Qwen3.6-35B-A3B 3-bit (unsloth UD): 255 tok/s, 16.8 GB, noul 0.72, choice 1.00 pooled, flip 4%.
- Both put >= 99.7% of probability on the option labels; both deterministic within a day.
- Gotchas: Gemma needs its empty thought block in the rendered text; Qwen needs
  `enable_thinking=False`. The HF Xet cache stores finished shards in a shared blob store, so
  `du` on a model dir after an interrupted download shows only stale `.incomplete` files.

## Unmeasured, needs a spike (original list; items 1, 2 and 4 now measured)
- M5 prefill tok/s and peak memory for a 3-bit Qwen3.6-35B-A3B on ~700-token prompts.
- Label single-token check in the Qwen3.6 tokenizer (letters, yes/no, scheme (b) digits).
- 3-bit vs full-precision agreement (only measurable with a GPU run or published numbers).
- Volume: about 20k teacher forwards x about 700 tokens = about 14M prompt tokens over Phases 4 and 6
  (DERIVED from the plan's decision counts, Choice labelled in 2 orders).

**Staleness:** free tiers change monthly; re-check Modal's Starter credit before relying on it.
