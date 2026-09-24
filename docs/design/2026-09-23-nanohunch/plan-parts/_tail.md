## Not doing (MVP), and what would bring each back

From `risks-overengineering.md` sections 5 and 6, plus ideas from the open
replications. Items with a trigger are on the later list; items without are
deleted. Anything added to the core from this list must still fit the
1,000-line budget.

| Item | Status | Trigger to add |
|---|---|---|
| Qwen3-4B training on a rented GPU (one Modal function, Volume checkpoints, `timeout=`) | later | Qwen3-4B B0 beats MiniCPM5-2B B0 by > 3 pts in Phase 3 (SemIf's published gap suggests it will), or a Mac epoch exceeds one night twice |
| HTTP `/v1/decide` server, Gradio demo, ZeroGPU Space | later | the release wants a live demo (after Phase 7) |
| Inference permutation pooling P = 2 and T per (type, P) | later | B0 flip rate > 30% in Phase 3, or trained top-1 agreement < 0.90 in Phase 6 |
| Order-invariant option masks (von `option_marker.py`: options cannot attend to each other, positions reset) | later, stretch | P = 2 pooling still leaves top-1 agreement < 0.90; it changes the format, so it is a v2 format (new ADR) |
| Many questions in one sequence with a block mask (decider `prompt.py`, reflex 4D mask) instead of trim branching | later, stretch | W1 latency matters for a demo; trim is simpler and is what the budget pays for |
| Official JevBench submission (sealed items, speed and cost axes) | later | the release is public; contact the maintainer per the JevBench README |
| Workflows 4 to 8, a whole held-out workflow, MMLU-Pro gold slice | later | target tier (40k decisions) after the MVP ships |
| Public dataset release with licence gating and secret scan | later | after the model release, if wanted |
| Real-text slice (GitHub issues/PRs) with PII scrub | later | a "real distribution" claim is wanted |
| MinHash near-duplicate scan, 3-seed variance study, contamination probe | later | real text added, or a headline delta within 1 pt of zero |
| Strong-tier (closed model) reference labels, B1 API baseline | later | a reviewer asks for a frontier comparison (never used for training, ADR-0004) |
| Jev API head-to-head (B2) | later, footnote only | Jev API access confirmed |
| Qwen3.5 hybrid branching, tree-mask packing, 0.8B latency variant | later, learning stretch | MVP shipped |
| Training on outputs distilled from Jev (von does this) | deleted | never: TypeSafe terms unverified, and it makes the release non-clean (ADR-0004) |
| Torch trainer and MLX/torch parity | deleted for MVP | returns only with the Modal item |
| Retry token bucket, circuit breaker, AIMD, SQLite state machine, HF checkpoint mirror, launchd watchdog, guard.py phase caps, runtime branch verification with fallback | deleted | none inside this project |

## Open questions

| # | Question | Who answers | By when | Blocks |
|---|---|---|---|---|
| Q1 | Do OpenRouter hosts for DeepSeek V4.1 Flash and Qwen3.6-35B-A3B return stable `top_logprobs` (>= 20) with thinking off? | P0-7 measurement | end of week 1 | Phase 4 teacher design |
| Q2 | Do Mac LoRA rates measured on synthetic weights hold on real MiniCPM5-2B weights over a 2 h soak? | P0-8 measurement | end of week 1 | Phase 5 training location |
| Q3 | What causes the 0.12 fp32 gap in MLX batched re-encode? | P0-6, 2 h time box | end of week 1 | batch size > 1 anywhere |
| Q4 | Is Qwen3-4B-Base B0 more than 3 pts better than MiniCPM5-2B-Base B0? | Phase 3 bake-off | week 4 | Modal later item |
| Q5 | Which pngwn card is right (dataset CC-BY-SA vs model card NC)? Treated as NC, eval-only, until answered | author asks on the HF discussion | before Phase 7 | nothing in MVP (eval-only either way) |
| Q6 | Is DeepSeek V4.1 Flash covered by the verified DeepSeek terms (4.2(3))? | author reads current ToS page | before Phase 4 bulk labelling | releasable-teacher allowlist |
| Q7 | Which host serves Qwen3.6-35B-A3B on OpenRouter, at what precision? | P0-7 logs host per row | end of week 1 | teacher pinning |
| Q8 | Does decider-2b's own inference code run on Apple Silicon (MPS) within about 30 min for our test set? It reports 133 ms per request on an M1 Pro | Phase 6 step 16a smoke test | week 8 | decider row; fallback is a one-off Modal run (about 1 USD) or dropping the row with a note |
| Q9 | Are the SemIf authored144 and JevBench public item schemas stable at the pinned commits? | Phase 3 step 9a (print keys first) | week 4 | external anchors |
| Q10 | Does our core fit in 1,000 lines once Phase 6 adds `bootstrap_ci` and `risk_coverage`? | `tools/loc.py` at each gate | every phase | if not, cut features before raising the budget |

## Assumption validation map

Assumption ids are from `_brief/requirements.md` (ledger), plus R-numbers from `risks.md`.

| Assumption | What it claims | Validated in | If wrong |
|---|---|---|---|
| A1 | teacher consensus agrees with a human on >= 75% of decisions | Phase 4 (100-item audit), Phase 6 (200-item audit) | drop that question type from the headline; single teacher + gold |
| A4 / A6 | teacher and GPU prices | Phase 0 (P0-7 cost log), `ledger/spend.csv` weekly | re-plan label counts; MVP has no GPU line |
| A5 | 10k to 15k decisions beat B0 by >= 3 pts | Phase 5 learning curve (kill rule at 3k), Phase 6 | pivot to the B0 + engine + calibration story; reflex's rejected adapters say this is a live risk |
| A7 | about 20 h/week | `ledger/hours.csv` weekly | calendar rules fire |
| A9 / A10 | 2B LoRA fits and trains on the M5 at >= 200 tok/s at 2k | Phase 0 (P0-8) | one Modal function (later item moves forward) |
| A11 | Mac W1 latency under about 1 s for MiniCPM5-2B | Phase 0 (stock), Phase 2 (own engine) | report honestly; latency is not the headline |
| A12 | permutation augmentation reaches >= 0.90 top-1 order agreement | Phase 5 pilot, Phase 6 | P = 2 pooling at inference, then von-style masks (v2 format) |
| A13 | training at >= 4k tokens avoids pngwn's long-input regression | Phase 6 length-bucket table | report it; that bucket becomes the ablation |
| A15 | Metal can wire about 16 to 19 GB | Phase 0 (MEASURED 19.07 GB working set) | smaller prefill chunks |
| M1 | a minimal core (<= 1,000 lines) is enough for a credible result | `tools/loc.py` every gate; Phase 6 open-replication table | publish the line count and the gap to decider-2b honestly; minimal is the claim, not "best" |
| X1 | our harness reproduces published numbers | Phase 3 SemIf sanity gate (within 5 pts of 0.686) | readout or format bug; fix before M1 |

## How to use this plan with an AI pair

Your convention (`rl-wordle/AGENTS.md`) is that you write the core and the
agent pairs. Copy that rule into this repo's `AGENTS.md` in Phase 0.

**You write:**
- the six core files: `fmt.py`, `engine.py`, `calibrate.py`, `train.py`,
  `dataset.py`, `evaluate.py`;
- `sample_spec` and `spec_questions` in `sources/triage.py`, because they define
  ground truth.

**Ask the agent for:**
- reviewing your diff against the phase's interface and tests;
- explaining an MLX or tokenizer behaviour you do not understand;
- walking through a file in `refs/` with you;
- writing glue in `sources/`, `label.py`, `cli.py`, `tools/`;
- proposing extra test cases;
- debugging a failing gate by reading logs;
- suggesting deletions when `tools/loc.py` says you are over budget.

A good prompt: "Here is my `fmt.render`. The test
`test_prefix_identical_across_questions` fails with <output>. Do not fix it;
tell me where my reasoning is wrong."

**Do not ask the agent for:**
- the body of any core function;
- a port of a reference repo's code into your core.

The stock tools in `skeleton/` are the one exception to the first rule: they
exist so your own code has a number to match.

## Re-assembling this file

```
cd docs/design/2026-09-23-nanohunch && python3 -c "import pathlib as P; d=P.Path('plan-parts'); parts=['phase-0-1.md','phase-2.md','phase-3.md','phase-4.md','phase-5.md','phase-6-7.md']; P.Path('plan.md').write_text((d/'_head.md').read_text().rstrip()+'\n\n'+'\n\n---\n\n'.join((d/p).read_text().strip() for p in parts)+'\n\n'+(d/'_tail.md').read_text())"
```
