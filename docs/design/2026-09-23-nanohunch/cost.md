# Cost model: nanohunch

Mode: GREENFIELD. Depth: standard. Author: sd-cost. Date: 2026-09-23.

> **Read this first.** Every price below is a snapshot. The per-token and
> per-hour prices marked MEASURED were fetched from the provider pages listed
> in section 0 on 2026-09-23. They change often (OpenRouter lists several
> model versions released within the last 60 days). The napkin-math anchors
> are order-of-magnitude only. **Re-check the exact page on the day you
> spend, and write the price you actually paid into the ledger (section 8).**
> Nothing here is a quote.

Labels: `MEASURED` = read from a named pricing page or API today, or published
by a named source. `DERIVED` = arithmetic on labelled inputs, arithmetic shown.
`ASSUMPTION` = a guess with a range.

Upstream files: `_brief/requirements.md` (cited `REQ:<line>`),
`_brief/system-map.md` (`MAP:<line>`), `_brief/research-notes.md` (`RN:<line>`).
`architecture.md` and `capacity.md` were empty templates when this was written,
so the candidate architectures in section 6 and the training throughput in
section 1.2 are this file's own assumptions (see `needs_from` in the manifest).

---

## Summary (the five numbers)

| Number | Value | Label |
|---|---|---|
| MVP cash, central (range) | **~72 USD (33 to 190)** | DERIVED, section 2 |
| Target cash, cumulative, central (range) | **~145 USD (61 to 358)** | DERIVED, section 2 |
| Reserve left at target vs the 200 USD cap | **~55 USD central; negative at the high end** | DERIVED |
| Bulk teacher labels for the whole target dataset | **~4 to 10 USD**, not the 60 USD allocated in `REQ:300` | DERIVED from MEASURED prices |
| Share of central target spend that is GPU (incl. debug, idle) | **~86 of 145 USD (59%)** | DERIVED |

Verdict: the design fits the cap at the central estimate with about a 25%
reserve. Only one thing breaks it: slow Qwen3.5 Gated DeltaNet training
kernels (`MAP:15-34`, `RN:137`) combined with the usual 2 to 3x debug
overhead. Labels are cheap. GPU time spent debugging is what decides whether
you finish under 200 USD.

---

## 0. Price sheet (what was fetched, when, from where)

*Why this matters:* a cost model is only as good as its input prices, and
2026 prices are not the ones in your memory or in `rl-wordle/PLAN.md`.

All fetched 2026-09-23.

### 0.1 GPU rental

| Provider / GPU | USD/h | Label | Source |
|---|---|---|---|
| RunPod A100 80 GB SXM, community / secure | 1.39 / 1.59 | MEASURED | https://www.runpod.io/pricing ("Updated September 13, 2026") |
| RunPod H100 SXM 80 GB, community / secure | 2.69 / 3.49 | MEASURED | same |
| RunPod RTX PRO 6000 96 GB, community / secure | 1.69 / 2.09 | MEASURED | same |
| RunPod L40S 48 GB, community / secure | 0.79 / 1.09 | MEASURED | same |
| RunPod RTX A6000 48 GB, community / secure | 0.33 / 0.53 | MEASURED | same |
| RunPod RTX 4090 24 GB, community / secure | 0.34 / 0.74 | MEASURED | same |
| RunPod storage: network volume / pod volume disk idle | 0.07 / 0.20 USD per GB-month | MEASURED | same |
| Modal A100 80 GB | 0.000694 USD/s = 2.50/h | MEASURED, per-hour DERIVED | https://modal.com/pricing |
| Modal H100 / L40S / RTX PRO 6000 / L4 | 3.95 / 1.95 / 3.03 / 0.80 per h | MEASURED per-second, DERIVED x3,600 | same |
| Modal CPU and memory (billed on top of GPU) | 0.0000131 USD/core/s (0.047/h), 0.00000222 USD/GiB/s (0.008/h) | MEASURED | same |
| Modal A100 all-in with 4 cores + 32 GiB | 2.50 + 4 x 0.047 + 32 x 0.008 = **2.94/h** | DERIVED | sizing is ASSUMPTION |
| Modal Starter plan free compute | **30 USD per month** | MEASURED | same (matches `MAP:181`) |
| Modal non-preemptible execution | 3x base price | MEASURED | same. Default is preemptible, so checkpointing is mandatory |
| Modal Volumes | 0.09 USD/GiB-month, first 1 TiB/month free | MEASURED | same |
| Modal environment-level budgets | Team plan (250 USD/month) only | MEASURED | same. Starter has no hard budget cap listed |
| Lambda 1x A100 SXM 40 GB / 1x H100 SXM / 1x A6000 | 1.99 / 4.29 / 1.09 | MEASURED | https://lambda.ai/pricing (plus tax) |

Lambda is 25 to 60% more expensive than RunPod for a single GPU, so it
drops out. Prices in section 2 use **RunPod A100 SXM secure, 1.59 USD/h**
as the basis unless stated. Section 4 covers the Modal free-credit lever.

### 0.2 Teacher LLM APIs (OpenRouter list prices, USD per 1M tokens)

Source for every row: https://openrouter.ai/api/v1/models, fetched
2026-09-23, fields `pricing.prompt`, `pricing.completion`,
`pricing.input_cache_read`, and `supported_parameters` (for `logprobs`).
Label: MEASURED.

| Model | In | Cache read | Out | Logprobs? | Role here |
|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0.10 | 0.01 | 0.50 | yes (+top_logprobs) | bulk teacher A, state generator |
| qwen/qwen3.6-35b-a3b (open weights, Apache-2.0 per HF API) | 0.15 | 0.05 | 1.00 | yes | bulk teacher B (releasable) |
| qwen/qwen3.8-27b (open weights, Apache-2.0, 27.8B params per HF API) | 0.42 | 0.085 | 3.00 | yes | alternative teacher B |
| qwen/qwen3.8-27b:free | 0 | n/a | 0 | **no** | useless at scale (50 req/day on free plan) |
| qwen/qwen3.7-flash | 0.03 | 0.006 | 0.13 | yes | cheapest state generator |
| openai/gpt-6-luna (":batch") | 0.10 (0.05) | n/a | 0.50 (0.25) | **no** | cheap non-logprob teacher, baseline B1 |
| openai/gpt-5.6-luna (":batch") | 0.20 (0.10) | 0.02 | 1.20 (0.60) | **no** | the "luna" in Jev's table (`RN:59`), superseded |
| openai/gpt-6-sol (":batch") | 2.00 (1.00) | n/a | 10.00 (5.00) | **no** | mid/strong reference option |
| anthropic/claude-opus-5.5 (":batch") | 4.00 (2.00) | 0.20 | 20.00 (10.00) | **no** | mid/strong reference option |
| openai/gpt-6-astra (":batch") | 10.00 (5.00) | n/a | 50.00 (25.00) | **no** | Jev's reference teacher 1 (`RN:49`) |
| anthropic/claude-fable-5.1 (":batch") | 10.00 (5.00) | 0.25 | 50.00 (25.00) | **no** | Jev's reference teacher 2 (`RN:49`) |
| openai/gpt-oss-20b | 0.018 | n/a | 0.09 | yes | very cheap, weak |

OpenRouter platform fee on the Standard pay-as-you-go plan: **5.5%** of
credits (MEASURED, https://openrouter.ai/pricing, 2026-09-23). The same page
lists "Budgets & Spend Controls" on Standard (MEASURED) and a 50 requests/day
limit on the Free plan (MEASURED). Direct provider APIs may be cheaper than
OpenRouter's list price: not fetched, ASSUMPTION.

Finding: **none of the closed strong-tier models expose logprobs** through
OpenRouter today. Every open-weight model on the list does. That single
column decides the labelling method (section 5, step S1).

### 0.3 Hosting and storage

| Item | Price | Label | Source |
|---|---|---|---|
| HF Space CPU Basic (2 vCPU, 16 GB) | free | MEASURED | https://huggingface.co/pricing |
| HF Space CPU Upgrade (8 vCPU, 32 GB) | 0.03 USD/h | MEASURED | same |
| HF Space T4 small / L4 / A10G small / A100 large | 0.40 / 0.80 / 1.00 / 2.50 USD/h | MEASURED | same |
| HF ZeroGPU (half RTX PRO 6000, 48 GB) hosting | free for personal accounts older than 30 days with a verified email, up to 2 Spaces | MEASURED | https://huggingface.co/docs/hub/spaces-zerogpu |
| ZeroGPU visitor quota: unauthenticated / free / PRO | 2 / 5 / 40 min per day | MEASURED | same |
| HF PRO | 9 USD/month | MEASURED | https://huggingface.co/pricing |
| HF Hub storage beyond included | 12 USD/TB-month public, 18 private; "egress and CDN included" | MEASURED | same. The free included quota is not stated on that page: ASSUMPTION that ~20 GB of public artefacts is free |
| HF community GPU grants | exist ("we also offer community GPU grants") | MEASURED existence; approval odds ASSUMPTION | same |

> TODO: unverified. The HF pricing page lists "Host ZeroGPU, Gradio & Docker
> Spaces" as a PRO perk, but the ZeroGPU docs say free accounts can host 2.
> Try it on account `ubermenchh` (`MAP:173`) before relying on it.

### 0.4 Not fetched (stay ASSUMPTION)

| Item | Assumed value | Range |
|---|---|---|
| Kaggle free GPU | ~30 GPU-h/week of T4x2 or P100 (16 GB) | 0 to 30 h; 16 GB is too small for 4B training at useful sequence lengths |
| Colab free GPU | T4 16 GB, unreliable sessions | same limitation |
| RunPod spot / interruptible discount | ~40 to 60% off on-demand | not on the pricing page today |
| Electricity price at the author's location | 0.20 USD/kWh | 0.08 to 0.30 |
| M5 MacBook package power under sustained MLX GPU load | 50 W | 30 to 70 W |
| Sales tax / GST on foreign digital services | 0% | 0 to 18% (18% if billed as an Indian resident, for example) |
| Provider minimum top-up that you cannot spend or refund | 5 to 10 USD per provider | 0 to 25 |

---

## 1. Line items

*Why this matters:* the lines you forget (idle GPU, debug reruns, cold
starts, fees, tax) are usually bigger than the lines you planned.

### 1.1 Workload quantities used everywhere

| Quantity | MVP | Target | Label / source |
|---|---|---|---|
| Train decisions | 15k | 40k | ASSUMPTION, `REQ:280-281` |
| Share of train decisions that needs teacher labels (the rest is public gold) | 60% = 9k | 60% = 24k | ASSUMPTION, `REQ:282` |
| Cal + test decisions (all teacher-labelled so consensus accuracy exists) | 2k + 5k = 7k | same 7k | `REQ:284-285` |
| Teacher-labelled decisions per teacher | 9k + 7k = **16k** | 24k + 7k = **31k** | DERIVED |
| Decisions per state | 3 (range 1.5 to 4) | 3 | ASSUMPTION, `REQ:283` |
| Synthetic states to generate (incl. 25% rejected) | (9k x 1 + 4.2k) / 3 x 1.25 = **~5.5k** | +5k x 1.25 = **+6.3k** | DERIVED. 4.2k = 60% of the 7k eval decisions |
| Tokens per teacher call, logprob method | 1,250 in (200 instructions + 1,000 state + 50 question), ~5 out | same | ASSUMPTION, state length from `REQ:163` |
| Tokens per teacher call, verbalized method (4 questions per call) | 1,300 in, 500 to 1,500 out (includes reasoning) | same | ASSUMPTION |
| Training tokens per epoch | 10M | 27M (scaled 40k/15k) | `REQ:287`, DERIVED |
| Epochs per full run | 2 | 2 | ASSUMPTION |

### 1.2 Training throughput (sd-capacity has not delivered; this is my own derivation)

LoRA with gradient checkpointing costs ~6N FLOPs per token (2N forward,
2N activation-gradient backward, 2N recompute). The frozen base needs no
weight gradients. ASSUMPTION.

| Base | N non-embedding | FLOP/token | A100 tok/s (good kernels, 30 to 40% MFU of 312 TFLOP/s) | A100 tok/s (slow DeltaNet fallback) | Label |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | ~0.5B (`REQ:151`) | 3e9 | 8k to 15k (overhead-bound; pngwn got 7.9k at 0.6B, `REQ:326-329`) | 3k to 6k | DERIVED / ASSUMPTION |
| MiniCPM5-2B (plain Llama, `MAP:35`) | 1.98B (`RN:76`) | 1.2e10 | 7.9k to 10.5k, use **9k** | n/a | DERIVED |
| Qwen3.5-4B | ~3.4B (`REQ:153`) | 2.0e10 | 4.7k to 6.2k, use **5k** | **700 to 2,300** | DERIVED; the fallback range is ASSUMPTION (`MAP:29-34`) |

Central for Qwen3.5-4B: **3,500 tok/s** at MVP and target (it assumes the
hub kernels load for most layers). Anchor: pngwn's whole arm B run was
68 min on one A100 (`RN:108`). That is 1.13 h x 1.59 = **1.80 USD** at today's
RunPod secure price (DERIVED).

### 1.3 MVP line-item table (cash USD, RunPod A100 secure basis, no free credits)

| # | Component | Quantity and arithmetic | Central | Range | Label |
|---|---|---|---|---|---|
| M1 | Phase 0 GPU benchmarks (A3 kernel speed on 2 bases, A9 memory sweep 1k/4k/8k/16k) | 1.5 h (1 to 2.5) x 1.59 | 2.4 | 1.6 to 4.0 | DERIVED |
| M2 | Teacher pilot: 200 decisions x 6 candidates (DS flash, Qwen3.6-35B, Qwen3.8-27B, luna, astra, fable) | cheap four < 0.1; strong two: 200 x 2 x 0.0048 to 0.011 | 3.3 | 2.0 to 4.5 | DERIVED |
| M3 | Synthetic state generation (DS V4.1 Flash) | 5.5k x (500 in x 0.10e-6 + 1,500 out x 0.50e-6 = 0.0008) | 4.4 | 2.5 to 8.6 (out 800 to 2,500 tok, plus 5% long states at 10k tok) | DERIVED |
| M4 | Bulk labels, teacher A: DS V4.1 Flash, logprob read, state prefix cached | 16k x 0.00008 (0.00005 cached to 0.00013 uncached, see 3.1) | 1.3 | 0.8 to 2.1 | DERIVED |
| M5 | Bulk labels, teacher B: Qwen3.6-35B-A3B, logprob read | 16k x 0.00014 (0.00010 to 0.00019) | 2.2 | 1.6 to 3.1 | DERIVED |
| M6 | Strong reference subset: 1,000 test decisions x (gpt-6-astra batch + claude-fable-5.1 batch), verbalized, ~1,000 out tokens per 4-question call | 1,000 x 2 x 0.0079 | 15.8 | 9.6 to 22 (500 to 1,500 out tokens) | DERIVED |
| M7 | Human audit, 250 decisions (`REQ:286`) | author time ~4 h | 0 | 0 | n/a |
| M8 | Pilot training runs, 0.8B/2B | 4 runs x 0.75 h x 1.59 | 4.8 | 2.1 to 8.0 | DERIVED |
| M9 | Full training runs, 4B, 2 clean runs | 2 x (20M tok / 3,500 tok/s = 1.6 h, then +50% for eval-in-loop, saving, and sequence padding = 2.4 h) x 1.59 | 7.6 | 3.1 to 25.4 (1.1 to 8 h per run) | DERIVED |
| M10 | **Failed and debug runs** (OOM, NaN, wrong loss mask, kernel fallback, bad data) | +1.5x of (M8 + M9), so training totals 2.5x clean | 18.6 | 5.2 to 66.8 (+1.0x to +2.0x) | ASSUMPTION multiplier per the brief |
| M11 | **Idle GPU, cold starts, image pulls, weight downloads** (9.3 GB per fresh box, `MAP:59`) | 20% of GPU lines M1 + M8 to M10 = 0.2 x 33.4 | 6.7 | 1.2 to 31 (10 to 30%) | ASSUMPTION |
| M12 | Mac electricity: ~10 eval passes + ~40 h of local debug | 10 x 0.045 kWh + 40 h x 50 W = 2.45 kWh, x 0.20 | 0.5 | 0.2 to 1.0 | DERIVED |
| M13 | Baselines: B1 gpt-6-luna, one call per question, 5k test decisions; B2 Jev | 5k x (1,250 x 0.10e-6 + 300 x 0.50e-6 = 0.000275) = 1.4; Jev 0.17 (`REQ:465`) | 1.6 | 1.0 to 2.0 | DERIVED |
| M14 | Storage: checkpoints as LoRA adapters on an HF private repo, labels on HF + laptop | < 2 GB total | 0 | 0 to 7 (if you park a 50 GB RunPod network volume for 2 months: 50 x 0.07 x 2) | DERIVED |
| M15 | API retries and parse failures | 5% of API lines (M2 to M6, M13 = 28.6) | 1.4 | 0.9 to 2.1 | ASSUMPTION |
| M16 | OpenRouter 5.5% platform fee | 0.055 x ~30 | 1.7 | 1.0 to 2.5 | DERIVED from MEASURED fee |
| M17 | Tax on foreign digital services | 0 to 18% of all cash | 0 | +0 to +13 | ASSUMPTION, excluded from central |
| M18 | Stranded prepaid minimums (RunPod, OpenRouter, one direct API) | not spent, still cash out | 0 | 0 to 25 | ASSUMPTION, excluded from central |
| M19 | Demo during build | local Gradio | 0 | 0 | DERIVED |
| | **MVP total** | | **72** | **33 to 190** | DERIVED |

Web-service lines that do not apply here, stated so nobody adds them back:
cross-AZ traffic, NAT gateway processing, and load balancers are 0 (no VPC,
single box). Internet egress is 0 because HF Hub includes egress and CDN
(MEASURED). The equivalent cost is GPU seconds spent downloading, which is in
M11. Observability SaaS is 0 (local logs plus the ledger). Support plans and
per-seat SaaS are 0 (Modal Starter has 3 free seats, MEASURED).

### 1.4 Target increment (on top of MVP)

| # | Component | Arithmetic | Central | Range | Label |
|---|---|---|---|---|---|
| T1 | More synthetic states | 6.3k x 0.0008 | 5.0 | 2.8 to 8.1 | DERIVED |
| T2 | More bulk labels, both teachers | 15k x (0.00008 + 0.00014) | 3.3 | 2.3 to 4.8 | DERIVED |
| T3 | Strong reference grown to 2,000 decisions (optional) | +1,000 x 2 x 0.0079 | 15.8 | 0 (skip) to 22 | DERIVED |
| T4 | Clean training: 2 full 4B runs at seq 8k (54M tokens each) + 2 ablation pilots (permutation augmentation, length) + 0.8B distilled variant | 2 x 54M / 3,500 tok/s = 2 x 4.3 h x 1.59 = 13.6; + 2 x 1 h x 1.59 = 3.2; + 1.5 h x 1.59 = 2.4 | 19.2 | 12.5 to 39.8 | DERIVED |
| T5 | Debug and failed runs (fewer than MVP; the loop already works) | +1.0x of T4 | 19.2 | 6.3 to 59.7 (+0.5x to +1.5x) | ASSUMPTION |
| T6 | Idle, cold start, downloads | 20% of (T4 + T5) | 7.7 | 1.9 to 29.9 | ASSUMPTION |
| T7 | Retries + OpenRouter fee on T1 to T3 | ~10% | 2.4 | 1.5 to 3.5 | DERIVED |
| T8 | Mac eval electricity | | 0.3 | 0.2 to 0.5 | DERIVED |
| T9 | Public demo, first month | ZeroGPU free | 0 | 0 to 9 (PRO) | MEASURED price |
| | **Target increment** | | **73** | **28 to 168** | DERIVED |

---

## 2. Three totals vs the 200 USD cap

*Why this matters:* a single number hides the tail. The high end tells you
which kill rule has to exist before you spend the first dollar.

| Scenario | Central | Range | Reserve left (200 minus central) | Meets `REQ:302` reserve >= 50? | Label |
|---|---|---|---|---|---|
| (1) MVP run | **72** | 33 to 190 | 128 | yes; also meets MVP <= 120 (`REQ:303`) | DERIVED |
| (2) Target run, cumulative | **145** | 61 to 358 | **55** | yes, barely | DERIVED |
| (2') Target with 18% tax | 171 | 72 to 422 | 29 | **no** | DERIVED, tax is ASSUMPTION |
| (2'') Target using Modal's free 30 USD/month for 3 months | **~97** | 40 to 310 | 103 | yes | DERIVED in section 4, lever L1 |
| (3a) "10x" = 10x target dataset (400k train decisions), same 4B recipe | 145 + ~290 = **~435** | 300 to 700 | **-235** | **no, breaks the cap about 2x** | DERIVED below |
| (3a') 10x dataset on MiniCPM5-2B, 1 epoch, self-hosted teacher B, Qwen3.7-flash generation | 145 + ~65 = **~210** | 170 to 290 | -10 | no, but close (fits if the target was done on Modal credits) | DERIVED below |
| (3b) "10x" = a second base model at target data size | 145 + **~16** = **~161** | 153 to 175 | 39 | fits the cap, fails the 50 reserve rule | DERIVED below |

Arithmetic for (3a), the increment over target: labels 240k teacher
decisions x 0.00022 (both API teachers) = 53. Generation 50k states x 1.25 x
0.0008 = 50. Training 2 epochs x 270M tokens / 3,500 tok/s = 43 h x 1.59 =
68 per run, x 2 runs = 137, +20% idle = 164. Strong reference unchanged at 0.
Retries and fees ~20. Total ~287.

Arithmetic for (3a'): self-hosted Qwen3.6-35B-A3B on a RunPod RTX PRO 6000
(96 GB fits the 72 GB bf16 weights, 36B params per HF API): 240k x 500
tokens = 120M tokens / 10k tok/s (ASSUMPTION, 5k to 15k for 3B-active MoE
prefill in vLLM) = 3.3 h x 2.09 = 7, plus ~3 fixed setup = 10. Teacher A via
API: 240k x 0.00008 = 19. Generation with Qwen3.7-flash: 62.5k x (500 x
0.03e-6 + 1,500 x 0.13e-6 = 0.00021) = 13. Training MiniCPM5-2B, 1 epoch:
270M / 9k tok/s = 8.3 h x 1.59 = 13, x 1.5 debug = 20. Total ~62 to 65.

Arithmetic for (3b): the label cache is model-agnostic, so new labels cost
0. MiniCPM5-2B: 54M tokens / 9k tok/s = 1.7 h x 1.59 = 2.7 per run; 2 runs x
2.5 debug x 1.2 idle = 16. The real cost is 12 to 20 h of MLX engine work
(section 6).

**Finding:** once the labels exist, a second base model is cheap (~16 USD). A
10x dataset only fits on the 2B dense path. The labelled dataset is the asset
that pays back, and it is also what you release.

### Ongoing: public demo hosting per month

| Option | USD/month | Arithmetic | Fits a 4B demo? | Label |
|---|---|---|---|---|
| Local Gradio on the Mac | 0 | | yes, W1 ~1.6 s (`REQ:191`) | DERIVED |
| **HF ZeroGPU Space (recommended)** | **0** | free hosting, up to 2 Spaces | yes, 48 GB slice; a free logged-in visitor gets 5 min/day, roughly 100+ W1 calls at 1 to 3 s each | MEASURED quota, DERIVED calls |
| HF CPU Basic Space | 0 | | only 0.8B is usable: ~10 to 30 s per W1 (ASSUMPTION, 2 vCPU at 50 to 150 GFLOP/s) | ASSUMPTION |
| HF PRO (more quota, 10 Spaces) | 9 | | yes | MEASURED |
| HF CPU Upgrade, always on | 21.9 | 0.03 x 730 h | 0.8B to 2B, slow | DERIVED |
| HF T4 small, always on / sleeping, 1 h/day awake | 292 / 12 | 0.40 x 730; 0.40 x 30 | yes | DERIVED |
| Modal web endpoint on L4, scale to zero, ~30 wakes x 5 min | ~2, inside the 30 free | 2.5 h x 0.80 | yes, 60 to 120 s cold start | DERIVED |

---

## 3. Unit economics

*Why this matters:* totals change when the plan changes. Unit costs tell you
what the next 10k labels, epoch, or eval will cost, and that is what you
decide with.

### 3.1 USD per labelled decision, per teacher option

Per-question call with the logprob method; per-state call (4 questions) with
the verbalized method. OpenRouter prices from section 0.2 (MEASURED). The
arithmetic is DERIVED.

| Teacher option | Method | Arithmetic (cached / uncached) | USD per decision | Per 10k decisions | Soft labels? |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | logprobs, 1 call per question, state prefix cached | cached: (1,250 x 0.10e-6 + 3 x (1,200 x 0.01e-6 + 50 x 0.10e-6)) / 4 + 5 x 0.5e-6; uncached: 1,250 x 0.10e-6 + 5 x 0.5e-6 | **0.00005 to 0.00013** | 0.5 to 1.3 | yes, from top_logprobs, no sampling |
| Qwen3.6-35B-A3B (API) | logprobs | same shape at 0.15 / 0.05 / 1.00 | **0.00010 to 0.00019** | 1.0 to 1.9 | yes |
| Qwen3.8-27B (API) | logprobs | same shape at 0.42 / 0.085 / 3.00 | 0.00024 to 0.00054 | 2.4 to 5.4 | yes |
| Qwen3.8-27B self-hosted, RunPod A100 secure, vLLM prompt logprobs | prefill only, ~375 to 700 tok/decision with prefix caching, 2k to 4k tok/s | variable: 500 / 3,000 x 1.59 / 3,600 = 0.000074; plus fixed ~5 USD (setup, 55.6 GB weights, vLLM start) | 0.00004 to 0.00016 variable; **0.0004 effective at 16k decisions** | 0.4 to 1.6 + fixed | yes, full distribution |
| gpt-6-luna batch | verbalized probabilities, 1 sample | (1,300 x 0.05e-6 + 600 x 0.25e-6) / 4 | 0.000054 | 0.54 | poor: verbalized probabilities are coarse |
| gpt-6-luna batch | k = 5 samples | 5 x 0.000054 | 0.00027 | 2.7 | yes, 0.2 granularity |
| gpt-6-sol batch | verbalized, ~1,000 out | (1,300 x 1e-6 + 1,000 x 5e-6) / 4 | 0.0016 | 16 | coarse |
| claude-opus-5.5 batch | verbalized | (1,300 x 2e-6 + 1,000 x 10e-6) / 4 | 0.0032 | 32 | coarse |
| gpt-6-astra batch or claude-fable-5.1 batch (Jev's reference pair) | verbalized, 500 to 1,500 out | (1,300 x 5e-6 + out x 25e-6) / 4 | **0.0048 to 0.011 each** | 48 to 110 each | coarse |
| Author (human audit) | hand label | 1 min per decision (`REQ:286`) | 0 USD, ~1 min | ~170 h | no |

Cross-check against Jev's table (`RN:59-60`): luna at 0.0033 USD per case,
spread over 10 to 40 questions, is 0.00008 to 0.0003 per question. The
0.00005 to 0.0003 range above agrees.

### 3.2 USD per training epoch, per base (RunPod A100 secure 1.59/h)

| Base | Epoch at 10M tokens (MVP) | Epoch at 27M tokens (target) | Label |
|---|---|---|---|
| Qwen3.5-0.8B | 10M / 10k tok/s = 0.28 h = **0.44** (0.30 to 0.88) | 1.2 | DERIVED |
| MiniCPM5-2B | 10M / 9k = 0.31 h = **0.49** (0.42 to 0.88) | 1.3 | DERIVED |
| Qwen3.5-4B, kernels load | 10M / 5k = 0.56 h = **0.88** | 2.4 | DERIVED |
| Qwen3.5-4B, central | 10M / 3.5k = 0.79 h = **1.26** | 3.4 | DERIVED |
| Qwen3.5-4B, DeltaNet fallback | 10M / 700 to 2,300 = 1.2 to 4 h = **1.9 to 6.3** | 5.2 to 17 | DERIVED from ASSUMPTION |
| Same on Modal A100 all-in (beyond free credit) | x 2.94 / 1.59 = **x 1.85** | | DERIVED |
| Same on RunPod RTX 4090 community (24 GB, seq <= 2k to 4k only, ~0.5x A100 speed, ASSUMPTION) | x 0.34 / (0.5 x 1.59) = **x 0.43** | not usable at 8k | DERIVED |

### 3.3 USD per full eval pass (5k test decisions, `REQ:213`)

| Where | Arithmetic | USD | Label |
|---|---|---|---|
| **Mac, 4B, MLX** | 3.1M tokens / 950 tok/s = 55 min; 0.92 h x 50 W = 0.045 kWh x 0.20 | **0.009** (0.002 to 0.03) | DERIVED |
| Mac, with the 4 extra permutations on Choice items (S3) | ~+30% tokens | 0.012 | DERIVED |
| A100 instead (forward only, 30% MFU: 94 TFLOP/s / 6.8 GFLOP/token = 14k tok/s) | 220 s + 5 min cold start = 8.7 min x 1.59 / 60 | 0.23 | DERIVED |
| Consensus labels for that test set (one-off, already in M4 to M6) | | 0 marginal | |

Evals belong on the Mac. They cost less than a cent, and running them there
also exercises the shipped engine.

### 3.4 Self-hosted inference on the Mac vs Jev's 0.042 USD per 1M input tokens (`RN:42-43`)

Electricity only. Hardware amortization is excluded because you already own
the Mac.

| Model | Seconds per 1M tokens | Wh | USD per 1M tokens at 0.10 / 0.20 / 0.30 per kWh | vs Jev 0.042 | Label |
|---|---|---|---|---|---|
| Qwen3.5-4B bf16, 950 tok/s, 50 W | 1,053 | 14.6 | 0.0015 / **0.0029** / 0.0044 | **~14x cheaper** (range 3.6x to 65x) | DERIVED from `REQ:153` and ASSUMPTION power |
| Qwen3.5-4B, pessimistic 500 tok/s, 70 W | 2,000 | 38.9 | 0.0039 / 0.0078 / 0.0117 | 3.6x to 11x | DERIVED |
| Qwen3.5-0.8B, 6,500 tok/s, 30 W | 154 | 1.3 | 0.00013 / 0.00026 / 0.00038 | **~160x cheaper** | DERIVED |

Caveat: Jev prices each input token once and runs at 70 to 500 ms. On the Mac
a 4B model takes ~1.6 s for W1. The marginal-cost claim is strong. The
latency claim only holds for the 0.8B variant (`REQ:193-196`).

---

## 4. Top 3 cost drivers and their levers

*Why this matters:* cutting a line that is 3% of spend is wasted effort.
Work on these three first.

Shares below are of the central target total (145 USD).

### D1. GPU training including debug reruns: ~70 USD (48%), plus idle ~14 USD

| Lever | Saves | What it costs you |
|---|---|---|
| **L1. Spend Modal's free 30 USD/month first** (MEASURED). Covers 30 / 2.94 = 10.2 A100-h per month. Central target needs 86 / 1.59 = 54 A100-h, so 3 months of credit covers 30 h and 24 h stay on RunPod (38 USD) | **~48 USD** (central target 145 to ~97) | Calendar pacing, because the credit is use-it-or-lose-it each month. Modal GPUs are preemptible (MEASURED; non-preemptible costs 3x), so checkpoint every <= 20 min, which `REQ:354` already requires. Past the free tier Modal costs 1.85x RunPod, so switch providers at the credit line. You already use Modal (`MAP:169-171`) |
| **L2. Phase 0 kernel gate, then MiniCPM5-2B if Qwen3.5-4B trains below ~2,000 tok/s** | Caps the 4B tail: high-end training drops from ~150 to ~30 | About 7 points of MMLU-Pro knowledge (70.8 vs 78.0, `RN:76`), and the MMLU-Pro slice target (`REQ:249`) gets harder. It also removes all DeltaNet state-branching work from the MLX engine (-10 to 20 h) |
| **L3. Debug on the Mac with a tiny config before any GPU run** (a random-init 2-layer model with the same code path; MiniCPM5 trains fast in mlx-lm, `MAP:262-263`) | Cuts the debug multiplier from 2.5x to ~1.7x: ~15 to 25 USD | 5 to 10 h to keep the training code device-agnostic (torch MPS/CPU for the smoke run). Bugs that only appear on CUDA (fla kernels, bf16 overflow) still slip through |
| L4. Pilots on RunPod RTX 4090 community (0.34/h) instead of A100 | ~3 USD on pilots | Community cloud reliability. 24 GB caps pilot sequences at 2 to 4k |
| L5. 1 epoch instead of 2 on full runs | ~half of T4 and M9: ~15 USD | Unknown quality cost, probably under 1 point on in-distribution items and more on calibration. Measure it on the 0.8B pilot first |
| L6. RunPod community (1.39) instead of secure (1.59) | 13% of GPU: ~11 USD | Hosts are less reliable, so there is more preemption-like loss |

### D2. Strong-tier reference labels: ~32 USD at target (22%)

| Lever | Saves | What it costs you |
|---|---|---|
| **Keep the subset at 1,000 decisions, not 2,000** | 15.8 | Accuracy CI vs the strong reference widens from +/- 1.8 to +/- 2.5 points (sqrt(0.16 / n) x 1.96). Fine for an MVP claim |
| Always use `:batch` variants (MEASURED at 50% of list) | already assumed; not using batch doubles D2 to ~63 | Up to 24 h turnaround (ASSUMPTION), so plan the job overnight |
| Use gpt-6-sol + claude-opus-5.5 batch instead of astra + fable | 1,000 x (0.0158 - 0.0048) = **11 per 1,000** | You lose the exact Jev reference pair (`RN:49`), so the Jev comparison becomes approximate |
| Drop the strong tier. Use the 250-decision human audit plus public gold labels | all of D2 | Weakest Jev comparability. The portfolio claim rests on gold and audit agreement, which is arguably more honest |

### D3. Invisible overhead (idle GPU, cold starts, downloads, stranded disks): ~14 USD central, ~70 USD tail

| Lever | Saves | What it costs you |
|---|---|---|
| **Run training as Modal functions, or as RunPod pods with a self-destruct timer** (section 8) | Removes the tail. One forgotten A100 weekend costs 48 h x 1.59 = **76 USD** (DERIVED), 38% of the whole cap | 3 to 6 h of wrapper scripting |
| Cache base weights on a Modal Volume (1 TiB free, MEASURED) or HF Hub, never on a stopped RunPod pod disk (0.20/GB-month idle, MEASURED) | 50 GB of stopped pod disk = 10 USD/month without a sound | Each session starts 1 to 3 min slower |
| Batch the GPU work into a few long sessions, not many short ones | ~half of cold-start minutes | Slower iteration |

Not a driver: **bulk teacher labels (~4 to 10 USD total)** and synthetic
state generation (~9 USD). `REQ:300` allocates up to 60 USD to teacher APIs.
Recommendation: reallocate it to ~35 (bulk plus generation ~15, strong
reference ~16, baselines and fees ~4) and move ~25 into the GPU and reserve
lines.

---

## 5. Step functions (treat as design constraints)

*Why this matters:* costs rarely grow smoothly. They jump at a boundary, and
knowing where the boundaries are is how you stay on the cheap side of each.

| # | Boundary | Below | Above | Label | Design consequence |
|---|---|---|---|---|---|
| S1 | **Teacher exposes logprobs or not** | 1 call gives the full distribution | k samples for a distribution (k = 5 is **5x**), or 1 verbalized sample with coarse, poorly calibrated probabilities | MEASURED: no closed strong model on OpenRouter supports `logprobs` today; every open-weight model listed does | Bulk soft labels come from open-weight models via API. Closed models provide the reference only |
| S2 | **Qwen3.5 DeltaNet kernels load on CUDA, or fall back** (`MAP:29-34`) | ~5k tok/s, a 4B epoch at 10M tokens ~0.9 USD | 700 to 2,300 tok/s, **2 to 7x** per epoch | ASSUMPTION until Phase 0 | Phase 0 gate (L2). This is the only step that can break the cap by itself |
| S3 | **GPU memory class**: 24 GB (seq <= 2 to 4k) / 48 GB (<= 8k) / 80 GB+ (<= 16k) (`REQ:457`) | RTX 4090 0.34 to 0.74/h | L40S 0.79 to 1.09; A100 1.39 to 1.59; RTX PRO 6000 96 GB 1.69 to 2.09 | MEASURED prices, memory ASSUMPTION | Cost per token is roughly flat across classes. The step is that the 8k target (`REQ:270`) rules out 24 GB cards, and 16k stretch rules out 48 GB |
| S4 | **Self-host teacher vs API** | API cheaper below the breakeven | self-host cheaper above | DERIVED: breakeven N = 5 USD fixed / (0.00014 - 0.00005) = **~55k decisions** | MVP and target (16k to 31k per teacher): use the API. 10x (240k): self-host. This also removes vLLM from the MVP learning list |
| S5 | **Modal free credit, 30 USD per calendar month** | free | 1.85x RunPod | MEASURED | Schedule GPU phases across month boundaries |
| S6 | **Batch vs synchronous API** | 50% price, hours of latency | 2x price, seconds | MEASURED prices, latency ASSUMPTION | Batch everything that is not interactive debugging |
| S7 | **Prompt-cache minimum prefix length** | short states pay full input price (2 to 10x more per decision for logprob calls) | cached reads at 10% (DS) to 33% (Qwen3.6) of input | cache prices MEASURED; minimum length (~1,024 tokens on some providers) ASSUMPTION | Put the state first and the question last in every teacher prompt so the prefix is shared |
| S8 | **Strong-tier reference size** | | +10 to 22 USD per extra 1,000 decisions | DERIVED | The subset size is fixed up front, from the CI you need |
| S9 | **Scalar cross-encoder (pngwn arm A) vs letter scorer** | 1 logits row per question | 1 forward row per option; p50 4 options, so **~4x training tokens** unless you pack shared prefixes; pngwn used a100x4 (`RN:105`) | DERIVED | At central prices a cross-encoder full run is ~4 x 7 = 28 USD clean, ~70 with debug. It fits once, not twice |
| S10 | **Tax** | 0% | +18% on everything | ASSUMPTION | At 18%, the central target leaves only a 29 USD reserve |
| S11 | **Kill rule at 60% spend = 120 USD before S2 MVP passes** (`REQ:437-438`) | continue | stop and pivot | requirement | A hard stop in the ledger guard (section 8) |
| S12 | HF ZeroGPU quota; PRO at 9 USD/month | free | 9/month | MEASURED | Keep the demo free unless visitors complain about the queue |

---

## 6. Cost of complexity (calendar, not dollars)

*Why this matters:* at 20 h/week your hours are scarcer than your dollars.
One extra week of calendar is worth more than 20 USD of GPU.

This is unpaid learning time, so no dollar rate is applied. Conversion used:
**20 h/week nominal, 12 h/week pessimistic realized** (`REQ:383`). Hours are
ASSUMPTION ranges for a solo author who writes the core by hand (`MAP:50-56`).

Candidates (my assumption; `architecture.md` was empty):

- **C0 Calibrated base, no training.** The untouched base is read through
  option-letter logits, then temperature-scaled. MLX engine with branching.
- **C1 Causal letter scorer on MiniCPM5-2B.** LoRA with candidate-row loss.
  Plain KV branching.
- **C2 Causal letter scorer on Qwen3.5-4B.** C1 plus DeltaNet state branching
  and kernel risk.
- **C3 Scalar cross-encoder on Qwen3.5-4B** (pngwn arm A style).

| Work package | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| Scaffold, ledger, spend guardrails | 6 to 10 | 6 to 10 | 6 to 10 | 6 to 10 |
| Public dataset adapters, split by state | 10 to 16 | 10 to 16 | 10 to 16 | 10 to 16 |
| Synthetic workflow generator | 8 to 14 (eval only) | 12 to 24 | 12 to 24 | 12 to 24 |
| Teacher labeller (cache, logprob read, consensus) | 10 to 16 | 12 to 20 | 12 to 20 | 12 to 20 |
| Human audit tool + audit | 6 to 10 | 6 to 10 | 6 to 10 | 6 to 10 |
| Eval harness (ECE, NLL, Brier, permutations, buckets, bootstrap, plots) | 20 to 30 | 20 to 30 | 20 to 30 | 20 to 30 |
| B0 reader + temperature scaling | 8 to 12 | 8 to 12 | 8 to 12 | 8 to 12 |
| LoRA from scratch, loss, training loop, checkpoint/resume | 0 | 20 to 35 | 20 to 35 | 25 to 40 (seq-cls head, per-option batching) |
| GPU job wrapper (Modal/RunPod) + guardrails | 0 | 6 to 12 | 6 to 12 | 6 to 12 |
| Running, debugging, analysing training runs | 0 | 15 to 30 | 20 to 40 | 25 to 45 |
| MLX engine with state branching | 12 to 20 (MiniCPM5) | 12 to 20 | 22 to 40 (DeltaNet conv + recurrent state) | 30 to 50 (per-option rows) |
| Parity tests MLX vs PyTorch | 6 to 10 | 6 to 10 | 8 to 12 | 8 to 12 |
| HTTP server + CLI | 4 to 8 | 4 to 8 | 4 to 8 | 4 to 8 |
| Model card, dataset card, report, demo Space | 6 to 10 | 10 to 16 | 10 to 16 | 10 to 16 |
| **Total hours** | **96 to 156** | **147 to 253** | **164 to 285** | **182 to 305** |
| **Calendar weeks at 20 h/week** | **4.8 to 7.8** | **7.4 to 12.7** | **8.2 to 14.3** | **9.1 to 15.3** |
| Calendar weeks at 12 h/week | 8 to 13 | 12 to 21 | 14 to 24 | 15 to 25 |
| Cash (target scope) | ~3 to 20 USD | ~95 to 125 | ~145 central | ~190 to 250 (breaks the cap with the debug multiplier) |

Ongoing operational load. There is no pager and no on-call (`REQ:386`).
Some things can spend money while you sleep: one rented pod or Modal app per
training run, and 2 to 3 API keys. Each needs the guard in section 8. New
runtimes to learn:

- C1: CUDA training on a rented box (familiar, `MAP:259-261`) and HF Spaces.
- C2: C1 plus flash-linear-attention kernels and DeltaNet state semantics.
- S4 above 55k decisions: vLLM as well.
- C3: C2 plus a sequence-classification head path in MLX, which does not
  exist today (`MAP:42-44`).

**Finding:** time binds before the budget does. The requirement "MVP in
<= 6 weeks, ~100 to 120 h" (`REQ:384`) matches only C0 (96 to 156 h). For C2,
an MVP subset (skip the 0.8B variant, the demo polish, and half the
workflows) is ~110 to 180 h, which is 5.5 to 9 weeks at 20 h/week. The
coupling cuts both ways: every extra calendar month also adds 30 USD of free
Modal compute.

---

## 7. The cheapest viable version

*Why this matters:* if a nearly free version teaches you most of the same
things, you should know that before you pay for the expensive one.

**C0: calibrated base, no training. Cash ~3 to 20 USD. 96 to 156 h.**

| Line | Arithmetic | USD |
|---|---|---|
| Synthetic states for cal + test only | 1.4k states x 1.25 x 0.0008 | 1.4 |
| Bulk labels, 2 cheap logprob teachers on 7k decisions | 7k x (0.00008 + 0.00014) | 1.5 |
| Strong reference, 500 decisions (optional) | 500 x 2 x 0.0079 | 0 to 7.9 |
| B0 zero-shot on 2 bases + temperature fit | Mac | < 0.1 |
| GPU | none | 0 |
| Fees, retries, B1, Jev | | ~2 |
| **Total** | | **~5 (3 to 13)** |

What it keeps: the MLX branching engine, the Jev-shaped server, the whole
eval harness, calibration evidence (T-scaling fixes most ECE, `RN:117-118`),
and a published dataset and eval report.

What it gives up:

- S1 by definition, because there is no delta over B0.
- Probably S3 (order consistency). The untouched letter reader has no
  permutation training. Test-time averaging over 2 to 4 permutations can
  recover part of it, at 2 to 4x branch cost.
- S4 is unaffected.
- No trained model to publish, and no hands-on distillation or LoRA learning.
  That is half the point of the project.

Is it within 20% of the trained design's cost? No. It is 10 to 30x cheaper,
and it does not meet S1. So it is a baseline, not a replacement.

**The alternative that does land near the design:** C1 on MiniCPM5-2B, with
no DeltaNet at all. Target cash is ~95 to 125 USD vs C2's ~145, which is 15 to
35% cheaper. It also removes 17 to 32 h and the only cap-breaking step (S2).
It meets every MVP SLI except possibly the MMLU-Pro-style slice (`REQ:249`,
>= 0.50), where it starts ~7 knowledge points behind. **Headline-grade
finding for the architect:** unless Phase 0 shows Qwen3.5-4B training at
>= ~3,500 tok/s on a rented A100, C1 is the cost-dominant choice. Qwen3.5-4B
is worth its premium only if the knowledge slice is the portfolio headline
(`REQ:479`, Q2).

Recommended build order, which also works as a budget ramp: build C0 first
for ~5 USD (it produces B0, the harness, and the engine every other
candidate needs). Then train C1 or C2 on top.

---

## 8. Spend guardrails (so a bug cannot blow the cap)

*Why this matters:* the realistic way to lose 76 USD is not a bad estimate.
It is a pod left running over a weekend, or a labelling loop that re-pays
for cached items.

| # | Mechanism | Where | Concrete setting |
|---|---|---|---|
| G1 | **Local ledger, append-only** | `spend/ledger.jsonl` in the repo (committed; no secrets) | One line per paid unit of work: `ts, provider, sku (model or GPU), units_in, units_out, seconds, unit_price_usd, price_source_url, price_fetched_date, cost_usd, run_id, git_sha`. `spend.csv` (S11, `REQ:423`) is reconciled weekly against provider invoices |
| G2 | **Single paid-call chokepoint** | `spend/guard.py`, which every API client and GPU launcher must call | `guard.reserve(estimate_usd, phase)` refuses if `cumulative + estimate > phase_cap`. Phase caps: Phase 0 = 10, MVP labels = 30, MVP GPU = 60, target labels = 30, target GPU = 60, hard total = 180 (20 kept back). Refuses outright past the 120 USD kill rule until S2 MVP is marked passed |
| G3 | **Dry-run estimator with explicit confirm** | every labelling or generation CLI | Prints `n_calls x est_tokens x price = X USD`. Needs `--confirm-usd X` within 20% of the estimate, otherwise it exits |
| G4 | **Content-addressed label cache** | `labels/cache/` keyed by `sha256(teacher_id, prompt)` (`REQ:356-357`) | A re-run never pays twice. Back it up off the laptop to an HF private dataset |
| G5 | **OpenRouter per-key credit limits** | OpenRouter dashboard ("Budgets & Spend Controls", MEASURED on Standard) | One key per phase with a limit equal to the G2 phase cap. Prepay only that phase's amount. Auto top-up off |
| G6 | **Prepaid-only GPU balance** | RunPod | Top up 25 USD at a time. No auto-reload. A zero balance is the final hard stop |
| G7 | **Trainer budget cap** | the training script | After 20 warm-up steps, measure `s_per_step` and compute `max_steps = floor(run_cap_usd x 3600 / (price_per_h x s_per_step))`, with `run_cap_usd = 20` (`REQ:340-341`). Abort if the planned steps exceed it. Log projected USD every 100 steps |
| G8 | **Provider-side timeouts** | Modal: `@app.function(timeout=...)` set to 1.3x the expected run, at most 24 h (Modal max, `MAP:185`). RunPod: at pod start run `(sleep $MAX_SECONDS; runpodctl remove pod $RUNPOD_POD_ID) &`, plus a `trap` in the training script that removes the pod on exit or error | The pod dies even if your laptop is closed. `runpodctl` and the `RUNPOD_POD_ID` env var are ASSUMPTION: verify on the first pod |
| G9 | **Watchdog on the Mac** | `launchd` job every 30 min | Queries the RunPod and Modal APIs for running GPUs. Sends a desktop notification if any has been up more than 1.2x its planned duration, and kills it at 2x |
| G10 | **No persistent pod disks** | RunPod | Terminate pods rather than stop them. Weights live on the HF Hub or a Modal Volume (free to 1 TiB). Stopped-pod disks bill 0.20/GB-month (MEASURED) |
| G11 | **Free hardware only on Spaces** | HF Space settings | CPU Basic or ZeroGPU. Never select paid hardware without a sleep timeout |
| G12 | **Weekly spend review** | calendar | Compare the ledger with the invoices. If they diverge by more than 5%, find the leak before spending more |

---

## 9. Assumptions that move the total most (for the planner)

*Why this matters:* these are the numbers to measure first. Each one can
move the total by more than 10 USD.

| Assumption | Central | Range | Moves target total by | Falsify with |
|---|---|---|---|---|
| Qwen3.5-4B LoRA training throughput on A100 | 3,500 tok/s | 700 to 6,200 | -25 to +150 USD | Phase 0 50-step run (~1 USD, M1) |
| Debug multiplier | 2.5x at MVP, 2.0x at target | 1.5x to 3x | -30 to +60 | ledger after the first two full runs |
| Idle/cold-start overhead | 20% | 10 to 30% | -10 to +15, tail +76 | G8 and G9 make it small |
| Strong reference size and output tokens | 2,000 decisions, 1,000 out | 500 to 2,000; 500 to 1,500 | -24 to +12 | decide the CI first |
| Tax | 0% | 0 to 18% | +0 to +26 | first invoice |
| Modal Starter free credit still active on this account | 30/month | 0 to 30 | -48 if used | `modal` dashboard (`MAP:272`) |
| Mean state length | 1,000 tokens | 400 to 2,000 | labels x0.4 to x2 (small), training x0.4 to x2 (large) | tokenize the first dataset build |
