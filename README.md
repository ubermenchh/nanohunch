# nanohunch

A small "System One" decision model, written by hand in MLX. Qwen3-4B plus a LoRA adapter reads a text state and one typed
question (choice, yes/no or score) and returns a calibrated probability for each answer, not text. The probabilities come
from the answer-label logits at the last token. No sampling, one forward pass per question.

This repo is a record of an experiment: what I tried to raise the model's score on a held-out decision benchmark,
with every number, what worked and what did not. The adapter is at
[huggingface.co/ubermenchh/nanohunch-decider-4b](https://huggingface.co/ubermenchh/nanohunch-decider-4b).

## Summary

- **Benchmark:** [OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions) (OD), 2,352 held-out questions, never trained on,
  no temperature fitted on it. Capability = (Intelligence + Calibration) / 2 from the dataset's own `score.py`.
  Its authors say differences under 2 points are not significant.
- **Best recipe: 57.1 mean Capability over four seeds (std 1.0)**, against 53.1 for the untrained Qwen3-4B-Base and 35.4 for untrained
  Qwen3-4B-Instruct. It is Instruct + LoRA on 4,000 decision-shaped teacher questions, trained toward the teacher's own probability
  (soft targets), with no temperature scaling.
- **What worked:** (1) training on decision-shaped questions, not public knowledge questions (+6 to +12 Intelligence over untrained
  Instruct, in all eight trained seeds); (2) soft targets, which make the model calibrated without post-hoc scaling and cut the seed spread from 3.5 to 1.0 std.
- **What did not:** public gold-label questions, 3x more data, only hard questions, rule-generated questions, a hard-heavy mix, a
  larger temperature-fitting set, a Brier-loss term, a higher learning rate, and a bigger model (Qwen3-8B).
- **Still far from the leaderboard:** the OD leaderboard lists 4B fine-tunes at 67.7 (Jeff-1.0-Qwen-4B) and 77.1 (Wald-4B v2.0). We are 10 to 20 points behind.
  Our remaining errors are concentrated in the hard tier (50% accuracy at 73% confidence).

## Try it: use the model from Hugging Face

The adapter is on Hugging Face in two formats: PEFT (`adapter_model.safetensors`, for transformers + peft) and MLX (`adapter.npz`, for this repo's code).

### Option 1: transformers + peft (no MLX, no clone)

Needs `pip install torch transformers peft`. The first run downloads Qwen3-4B-Instruct-2507 (about 8 GB) and the adapter (132 MB). The PEFT file was converted
from the MLX one with [to_peft.py](to_peft.py) and checked against it on 40 benchmark questions: the same top answer on 40 of 40, probabilities within 0.022.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, ADAPTER = "Qwen/Qwen3-4B-Instruct-2507", "ubermenchh/nanohunch-decider-4b"
tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16), ADAPTER).eval()


def ask(state, question, options):
    letters = "ABCDEFGH"
    prompt = "\n".join([state, "", f"Question: {question}", *[f"{letters[i]}. {o}" for i, o in enumerate(options)], "Answer:"])
    ids = tok(prompt, return_tensors="pt").input_ids
    labels = [tok.encode(f" {letters[i]}")[0] for i in range(len(options))]
    with torch.no_grad():
        logits = model(ids).logits[0, -1, labels].float()
    return dict(zip(options, torch.softmax(logits, -1).tolist()))


# options are written "name: description"; a yes/no question has the two options "true: ..." and "false: ...";
# a score question has levels "0: ...", "1: ...", lowest first
state = "Customer: my parcel arrived broken and I want my money back today."
print(ask(state, "What does the customer want?", ["refund: money back", "replacement: a new parcel", "info: just asking something"]))
print(ask(state, "Should this ticket be escalated to a human agent?", ["true: The proposition is true.", "false: The proposition is false."]))
```

Options are written `name: description`. A yes/no question has the two options `true: ...` and `false: ...`; a score question has levels `0: ...`, `1: ...`, lowest first.
Output: `refund 0.962, replacement 0.021, info 0.017`, then `true 0.593, false 0.407`. Use the logits as they are (temperature 1). The snippet ran on a CPU.

### Option 2: this repo (MLX)

It needs an Apple Silicon Mac (it ran on a 24 GB Mac; the model needs about 8 GB) or a Linux machine with `mlx[cuda12]`. It reads the typed question format directly, so you do not write the option text yourself.

```bash
git clone https://github.com/ubermenchh/nanohunch && cd nanohunch
uv sync
uv run python ask.py        # runs three example questions
```

Example output (probabilities per answer; for yes/no, `true` is "yes"; for score, the keys are the levels 0, 1, 2, ...):

```
{'chitchat': 0.013, 'extraction_json': 0.978, 'data_analysis': 0.009}
{'true': 0.594, 'false': 0.406}
{'0': 0.367, '1': 0.538, '2': 0.045, '3': 0.036, '4': 0.014}
```

**Ask your own question.** [ask.py](ask.py) loads the adapter once and exposes `ask(state, question)`. Open it, edit the examples at the bottom, or
`from ask import ask` in your own script (importing it loads the model). A question is a dict with one of three types:

```python
state = "Customer: my parcel arrived broken and I want my money back today."

# choice: pick one named option
ask(state, {"type": "choice", "instructions": "What does the customer want?",
            "criteria": {"refund": "money back", "replacement": "a new parcel", "info": "just asking something"}})

# noul: a yes/no question (criteria can be None, or {"true": "...", "false": "..."} to describe each answer)
ask(state, {"type": "noul", "instructions": "Should this ticket be escalated to a human agent?"})

# score: levels from lowest to highest, 2 to 10 of them
ask(state, {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "somewhat", "urgent", "critical"]})
```

- `state` can be text, or a JSON-like dict or list (it is written out as JSON).
- Up to 8 options per question. Probabilities are for the options in the order you give them; answers can shift a little when you reorder
  them (the "flip rate" in Appendix A), so reorder and average if you need a steadier number.
- The model reads the answer once, at the last token, so it is fast (one forward pass) and has no text output or explanation.
- The temperatures in `config.json` are all 1 on purpose (see [the soft-target section](#the-details-behind-the-two-solid-results)).
- To see what the adapter adds, comment out the `model.load_weights(...)` line in `ask.py` to get the plain Instruct model, and compare.
- To score a whole file of questions the way the benchmark does, use `od.py` (see [Reproduce](#reproduce)).

Treat it as a research artifact: 57.1 on OpenDecisions is above the untrained base model but far below the leaderboard, and it is weakest on hard,
trap-style questions (see [Where the remaining error is](#where-the-remaining-error-is)).

## Results

Capability on OpenDecisions. "Seeds" are training seeds; the mean and std are over those seeds. The last column compares with the
untrained base model, 53.1.

| Recipe | Seeds | Capability per seed | Mean (std) | vs base |
|---|---|---|---|---|
| Qwen3-4B-Base, untrained | n/a | 53.1 | 53.1 | 0 |
| Qwen3-4B-Instruct-2507, untrained | n/a | 35.4 | 35.4 | -17.7 |
| Base + LoRA on public questions (BoolQ, ARC, CSQA) | 1 | 51.4 | 51.4 | -1.7 |
| Instruct + decider data, hard labels, fitted temperature | 4 | 56.7, 54.1, 48.5, 52.2 | 52.9 (3.5) | -0.2 |
| &nbsp;&nbsp;same, probabilities of the four seeds averaged | | | 55.9 | +2.8 |
| Instruct + decider data, soft targets, fitted temperature | 4 | 54.6, 55.7, 52.0, 54.3 | 54.2 (1.6) | +1.1 |
| **Instruct + decider data, soft targets, temperature 1** | **4** | **57.5, 58.3, 56.0, 56.4** | **57.1 (1.0)** | **+4.0** |

Full per-seed numbers, including Intelligence, Calibration and accuracy by question type:

| Model | Capability | Intelligence | Calibration | choice | yes/no | score |
|---|---|---|---|---|---|---|
| Qwen3-4B-Base, no training | 53.1 | 26.3 | 80.0 | 55.4% | 68.9% | 40.8% |
| Base + LoRA on BoolQ/ARC/CSQA (n900) | 51.4 | 24.4 | 78.4 | 55.5% | 67.6% | 40.3% |
| Qwen3-4B-Instruct-2507, no training | 35.4 | 31.0 | 39.8 | 62.4% | 69.6% | 45.7% |
| Hard labels, fitted T, seed 0 | 56.7 | 42.8 | 70.7 | 68.3% | 74.4% | 57.1% |
| Hard labels, fitted T, seed 1 | 54.1 | 40.3 | 67.8 | 67.3% | 73.0% | 53.9% |
| Hard labels, fitted T, seed 2 | 48.5 | 36.7 | 60.3 | 66.6% | 71.8% | 50.3% |
| Hard labels, fitted T, seed 3 | 52.2 | 39.6 | 64.8 | 67.1% | 73.7% | 51.0% |
| Soft targets, fitted T, seed 0 (Modal) | 54.6 | 40.5 | 68.8 | 68.3% | 73.2% | 54.1% |
| Soft targets, fitted T, seed 1 (Modal) | 55.7 | 43.2 | 68.1 | 68.9% | 75.1% | 55.3% |
| Soft targets, fitted T, seed 2 | 52.0 | 39.7 | 64.3 | 66.8% | 73.4% | 56.0% |
| Soft targets, fitted T, seed 3 | 54.3 | 40.7 | 67.9 | 66.6% | 74.2% | 53.1% |
| **Soft targets, T = 1, seed 0 (Modal)** | **57.5** | 40.3 | 74.6 | 68.3% | 73.2% | 54.1% |
| **Soft targets, T = 1, seed 1 (Modal)** | **58.3** | 43.0 | 73.6 | 68.9% | 75.1% | 55.3% |
| **Soft targets, T = 1, seed 2** | **56.0** | 39.3 | 72.7 | 66.8% | 73.4% | 56.0% |
| **Soft targets, T = 1, seed 3** | **56.4** | 40.6 | 72.3 | 66.6% | 74.2% | 53.1% |

Accuracy by type is the same with or without temperature scaling, because a temperature never changes which answer is largest. The hard-label seed 1 number
is from rescaling saved predictions to per-type temperatures, which gives the same probabilities up to numerics.
"Modal" means the run was trained on a Modal H100 with MLX's CUDA backend, the others on an Apple Silicon Mac. The Modal-trained
adapter loads on the Mac and gives the same probabilities to within 0.04.

## Ablations: what worked and what did not

Every row changes one thing from the baseline named in the last column. All use the setup in [Method](#method). Differences under
about 3 points are inside the seed spread of the hard-label recipe (std 3.5), so only the first two rows are solid results.

| Change | Result (Capability) | Baseline for comparison | Verdict |
|---|---|---|---|
| Train on decision-shaped teacher questions instead of public ones | Intelligence 31.0 to 36.7-43.2 across the 8 seeds; accuracy +2 to +12 points by question type | Untrained Instruct | **Worked.** Solid in all 8 trained seeds |
| Soft targets (`--soft`) | 54.2 (std 1.6) | 52.9 (std 3.5) hard labels, same temperature fit | Small gain, and the seed spread halves |
| Soft targets, no temperature scaling | 57.1 (std 1.0) | 54.2 soft targets with fitted T | **Worked.** +2.9, and +4.2 over hard labels |
| One temperature per question type, instead of one | +1.9 (seed 0: 54.8 to 56.7), +1.2 (seed 1: 52.9 to 54.1) | One temperature | Small. I chose it after seeing an OD score, so it is optimistic |
| Fit the temperature on random held-out decider questions | 50.1 (seed 0) | 54.8, fitted on hard held-out questions | Hurts: those questions are easy (median teacher probability .95) |
| Fit the temperature on 1,000 instead of 300 hard questions | 56.3, 54.0, 48.3, 52.6 (hard labels) | 56.7, 54.1, 48.5, 52.2 | No change (within 0.5). The temperature fit is not the source of the seed spread |
| 12,000 teacher questions instead of 4,000 | 55.0 | 56.7 (seed 0) | No gain. Accuracy flat, calibration lower, training loss 0.025 (over-fit to easy questions) |
| Only the 1,850 hard teacher questions (teacher probability under .8) | 51.4 | 54.8 (one temperature, 4,000 q) | Worse. Calibration lowest of all runs; fitted T .90 does not transfer |
| 2,000 teacher + 2,000 rule-generated questions | 54.2 | 56.7 (seed 0) | No measurable gain (within seed spread), with half the teacher data |
| Hard-heavy mix: 1,500 hard + 2,500 easy, soft targets | 49.5, 50.6 | 54.6, 55.7 (soft, fitted T, same seeds) | Worse. Intelligence unchanged; hard-tier accuracy 49.6% against 50.8% |
| Qwen3-8B instead of the 4B, soft targets, T = 1 | 55.1, 57.0 | 57.5, 58.3 (4B, same seeds) | No gain. Intelligence about 42 for both |
| Cross-entropy + 0.5 x Brier loss (public data) | NLL -.002 [-.003, -.000] | Cross-entropy | No real gain. Reverted (commit `33ad703`) |
| 900 questions per dataset, learning rate 2e-4 (public data) | Accuracy within noise of the untrained model | 350 questions, lr 5e-5 | No gain. See Appendix A |
| Hardest public questions only (`run.py --hard`) | Accuracy within noise; calibration worse | Random 350 | No gain. See Appendix A |

### The details behind the two solid results

**Decision-shaped data.** Training on public gold-label questions (BoolQ, ARC-Challenge, CommonsenseQA) never moved OD: three adapters
scored within 1.3 points of the base model. Those questions test knowledge; OD tests deciding between plausible options under a typed
rule. Training on the decider teacher data (below) changed that.

**Soft targets and the temperature.** Each decider question comes with `teacher_p`, the teacher's probability for the gold answer.
The soft target gives the gold option `p` and spreads the remaining `1 - p` evenly over the other options. A temperature fitted on
held-out decider questions then came out *below* 1 for soft-target models (0.65 to 1.1, mostly under 1): on easy decider questions the soft
model is under-confident, so the fit sharpens it, and on the much harder OD questions that made it over-confident. Undoing the
temperature (`od.py --temp 1`) raised Capability by 2.1 to 4.7 points in all six soft-target runs (four 4B seeds, two 8B seeds). I found this on seeds 0 and 1,
then tested it on seeds 2 and 3, which it also improved (52.0 to 56.0 and 54.3 to 56.4). Hard-label models still need a fitted
temperature, so the fair comparison is soft targets without one against hard labels with one.

### Where the remaining error is

OD questions carry a tier, a family and a "trap" answer (a plausible wrong option). The table uses the four soft-target 4B seeds' averaged
predictions, with the fitted temperatures (this analysis was done before the temperature finding, so confidences are slightly high).

| Tier | Questions | Accuracy | Mean confidence | Picks the trap answer |
|---|---|---|---|---|
| Easy | 443 | 87.1% | 0.87 | 2.7% |
| Standard | 407 | 74.7% | 0.79 | 14.5% |
| Judge | 581 | 71.9% | 0.75 | 12.7% |
| **Hard** | **921** | **49.6%** | **0.73** | **30.5%** |

- The model is well calibrated on easy questions and over-confident on the hard tier: right half the time, 73% sure, and in a third of cases
  it picks the trap.
- Weak families (accuracy): `usecase` 42%, `fairness` 52%, `jaggedness` 52%, `perturb_pair` 57%, `multilingual` 64%. Strong: `llm_router` 77%, `negation_pair_gen` 99%.
- Weak use cases: insurance claims, financial crime, risk assessment, LLM guardrails (about 50%). Many non-English languages score below English (67%), for example Spanish 56% and Arabic 52%, but with only 8 to 46 questions per language.
- The decider training data already covers these domains (by keyword match: about 22% of its 3,043 states mention finance or fraud, 9% insurance, 10% legal, 42% support), but mostly as easy questions:
  only 10% of its questions are hard. Adding more hard questions did not help (rows above).

## Method

**Model and readout.** Prompt = the state, then `Question: ...`, then options labelled `A.` to `H.` (at most 8), then `Answer:`. The logits of the label
tokens at the last position, taken from the output head, go through a softmax over the options: a restricted softmax, one forward
pass. Yes/no questions are two options (true/false), score questions are levels 0..n. Option order is shuffled during training
and the gold label remapped.

**Training.** LoRA on all seven projections (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`) of all 36 layers, rank 16, alpha 32
(alpha/rank = 2), AdamW, learning rate 1e-4 with 10% warmup and cosine decay, 32 questions per optimizer step, one pass. Start from `Qwen/Qwen3-4B-Instruct-2507`.
Loss is cross-entropy over the answer labels; with `--soft`, cross-entropy against the soft target. Data is 4,000 teacher questions plus
810 public replay questions (270 each from BoolQ, ARC-Challenge, CommonsenseQA), 150 steps.

**Teacher data.** [decider](https://github.com/Mapika/decider) `custom_questions.jsonl` (Apache-2.0, pinned commit `91b6fa9`): 23,392 typed
questions (8,875 yes/no, 8,505 choice, 6,012 score), each with a `teacher_p`. I keep the 22,514 with `teacher_p >= 0.5`;
2,152 of those are "hard" (`teacher_p < 0.8`). Held-out questions after the trained ones are used to fit temperatures.

**Evaluation.** `od.py` scores a model or adapter with one pass per question, options in their given order, as on the leaderboard.
Predictions are scored by the dataset's own `score.py` (stdlib only). OD is 2,352 questions: 1,178 choice, 586 yes/no, 588 score.
Nothing is trained on OD and no temperature is fitted on it.

**Code.** Everything is a few flat files, written by hand.

| File | What it does |
|---|---|
| `model.py` | Load the model, LoRA layers, read the answer-label logits at the last token |
| `data.py` | Question to prompt, option shuffling with the gold remap, public datasets, decider loader |
| `train.py` | Loss over the answer labels (hard or soft target) and the training loop |
| `eval.py` | Pooling over option orders, accuracy, flip rate, ECE, temperature fit |
| `run.py` | Phase 1: the public-data experiment in `# %%` cells |
| `run_decider.py` | Phase 2: train on decider's teacher questions (`--soft`, `--hard`, `--rules`) plus public replay |
| `ask.py` | Ask the adapter from Hugging Face a typed question (a runnable example) |
| `to_peft.py` | Convert an adapter from MLX `.npz` to PEFT format |
| `od.py` | Score a model or adapter on OpenDecisions (`--temp 1` to ignore the fitted temperatures) |
| `refit_temp.py`, `retemp.py` | Fit one temperature per question type on hard held-out questions; rescale saved predictions offline |
| `make_rules.py` | Generate rule-conditioned questions with decider's generator |
| `analyze.py` | Paired-bootstrap intervals for phase 1 |

## Problems found along the way

- **LoRA weights in float32 promoted the activations.** All-layer LoRA took 24.8 GB and 45 s per step. Casting the LoRA output back to the input dtype
  gave 11.6 GB and 2.1 s per step (the same model, a 20x speed-up).
- **The output head was read from the embedding matrix.** That is right for Qwen3-4B (tied embeddings) but not Qwen3-8B, which has a separate `lm_head`.
  The untrained 8B scored Capability 6.5 (choice accuracy 26%, chance level) until `answer_logits` read `lm_head` when it exists. The 4B results are unaffected.
- **A temperature fitted on easy held-out questions does not transfer** to a benchmark of harder questions (see the soft-target section). Fitting on
  random held-out questions gave 50.1; on hard ones, 54.8.
- **A single seed can mislead.** Seed 0 of the hard-label recipe scored 56.7 and looked like a clear win; the four seeds range from 48.5 to 56.7.
  I wrote up the "best" adapter first and corrected it after the other seeds.
- **Modal detached runs.** `modal run --detach` with a `.map` over jobs cancelled all but the last job when the local process disconnected.
  The fix is `modal deploy` plus `Function.spawn`.

## Limits

- **Four seeds, small effects.** The soft-target gain (+4.2 over hard labels) is about 2.3 standard errors of the difference of the two means (std 3.5 and 1.0, four seeds each), and those stds are themselves estimated from four runs. The 8B and hard-heavy results have two seeds each.
- **Choices made after seeing OD scores.** Per-type temperatures, and temperature 1, were chosen after looking at OD results. Seeds 2 and 3 were an out-of-sample check
  for the temperature-1 choice, and it held, but the recipe has still been tuned against this benchmark. Treat 57.1 as optimistic.
- **Mixed platforms.** Some seeds were trained on a Mac, some on CUDA. Results differ by about 1 point from numerics, and the soft T=1 comparison
  mixes both.
- **Calibration is still below the base model** (72 to 75 against 80). The untrained base is not accurate, just uniformly unsure, which scores well on calibration.
- **One benchmark, one prompt format.** One pass per question, options in the given order. Not tested on other formats or longer states.
- **No regression check** on the public datasets for the final adapter.

## Reproduce

```bash
uv sync
uv run python run_decider.py --soft --seed 0                       # train; writes adapter_decider_soft_n4000.npz + .json
uv run python od.py --adapter adapter_decider_soft_n4000.npz --temp 1   # writes runs/od_adapter_decider_soft_n4000.jsonl
python3 -I score.py runs/od_adapter_decider_soft_n4000.jsonl       # score.py comes from the OpenDecisions dataset repo
```

Add `--small` to `run_decider.py` for a quick check. Phase 1: `uv run python run.py` (`--small`, `--seed S`, `--n N`, `--hard`). Both log to wandb unless `--no-wandb` is given to `run_decider.py`.
The Modal runs used `mlx[cuda12]==0.32.4` on an `nvidia/cuda:12.9.1-devel-ubuntu22.04` image with one H100 per run (4B: about 35 minutes for
training, refit and scoring; 8B training 10 to 15 minutes); those launch scripts are not in the repo.

## References

- **OpenDecisions** benchmark and leaderboard (`score.py`, MIT; data CC BY-SA 4.0): https://huggingface.co/datasets/jgeuter/OpenDecisions
- **decider** teacher data and generators (Apache-2.0), pinned at commit `91b6fa9b94ab6c4dc2c63d3b14d413243b4fffcc`: https://github.com/Mapika/decider
- Yang et al. **Qwen3 Technical Report**, 2025. https://arxiv.org/abs/2505.09388
- Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models**, 2021. https://arxiv.org/abs/2106.09685
- Guo, Pleiss, Sun, Weinberger. **On Calibration of Modern Neural Networks** (temperature scaling), ICML 2017. https://arxiv.org/abs/1706.04599
- Hinton, Vinyals, Dean. **Distilling the Knowledge in a Neural Network** (training toward a teacher's probabilities), 2015. https://arxiv.org/abs/1503.02531
- Brier, G. W. **Verification of forecasts expressed in terms of probability**, Monthly Weather Review, 1950 (the Brier score used in the dropped loss experiment).
- Clark et al. **BoolQ**, 2019. https://arxiv.org/abs/1905.10044
- Clark et al. **Think you have Solved Question Answering? Try ARC**, 2018. https://arxiv.org/abs/1803.05457
- Talmor et al. **CommonsenseQA**, 2019. https://arxiv.org/abs/1811.00937
- **MLX** and **mlx-lm**: https://github.com/ml-explore/mlx, https://github.com/ml-explore/mlx-lm
- Karpathy. **nanoGPT**, the style this repo follows (few flat files, no frameworks): https://github.com/karpathy/nanoGPT

## Appendix A: phase 1, the public datasets

Setup: Qwen3-4B-Base, LoRA (rank 8, alpha 160, 8 layers, lr 5e-5), trained on 1,050 public gold questions
(350 each from BoolQ, ARC-Challenge and CommonsenseQA train splits) with the option order shuffled each time.
The temperature is fitted on 600 other train-split questions (200 each), never on training or validation
questions. Scores are on 750 validation questions (250 each). Seed 0 unless said otherwise; the same seed gives the
same result, checked twice. Seeds 1 and 2 are below. Intervals are 95%, from a paired bootstrap that resamples the questions (`analyze.py`).

**The base model is already strong on this data.** B0 accuracy is .892 (BoolQ), .896 (ARC), .820 (CSQA).

**Trained minus B0, seed 0, all 750 questions** (a negative number is better for NLL, ECE and flip):

| | Accuracy | NLL | ECE | Flip rate |
|---|---|---|---|---|
| Raw probabilities | 0.000 [-.009, +.009] | -.082 [-.095, -.069] | -.086 [-.093, -.067] | -.025 [-.044, -.007] |
| Each model with its own temperature | +.001 [-.008, +.011] | -.014 [-.024, -.005] | -.011 [-.024, +.008] | -.025 [-.044, -.007] |

- Accuracy does not change, to within about one point.
- Training makes the raw probabilities much better calibrated. Most of that gain is also available
  from temperature scaling alone (B0 ECE falls from .065 / .149 / .180 to .019 / .054 / .045).
- After scaling, a small gain in loss remains for seed 0, and answers change less when the options are
  reordered. Seeds 1 and 2 (next table) show that the order effect is mostly seed luck.
- B0 is under-confident (temperature .554), not over-confident. Pooling the probabilities over three
  option orders softens them.

**Three training seeds** (350 questions per dataset, 750 validation questions, each model with its own temperature):

| | Temperature | Accuracy | NLL | Flip rate | CSQA flip rate |
|---|---|---|---|---|---|
| B0 (no training) | .554 | .867 | .361 | .191 | .316 |
| Seed 0 | .841 | .868 | .347 | .165 | .248 |
| Seed 1 | .841 | .871 | .350 | .171 | .284 |
| Seed 2 | .766 | .868 | .356 | .188 | .312 |

- Accuracy does not change in any seed.
- The loss gain after temperature scaling is small (.005 to .014) and one seed is barely better than B0.
- The CSQA flip rate falls by anything from 0 to .07 depending on the seed. Do not read the seed 0 drop
  as a result.
- Training seed alone moves the scores by more than the bootstrap interval suggests. Seed 2 minus seed 0:
  NLL +.009 [+.001, +.016], flip +.023 [+.005, +.041]. Raw NLL differs by up to .016 between seeds.

**Tried and dropped: cross-entropy plus 0.5 x Brier loss** (commit `33ad703`, reverted after it).
Brier minus cross-entropy, both with their own temperature, all 750 questions: accuracy 0.000,
NLL -.002 [-.003, -.000], ECE -.002 [-.011, +.010], flip -.003 [-.008, +.003]. The NLL difference is
smaller than the spread between cross-entropy runs with different seeds (up to .009), so it is not a real gain. Plain cross-entropy with a held-out temperature is enough.

**More data and a higher learning rate did not help either.** Both runs below changed one thing from
the run above (350 questions per dataset, learning rate 5e-5), all 750 validation questions, each
model with its own temperature:

| Change | Accuracy | NLL | ECE | Flip rate |
|---|---|---|---|---|
| 900 questions per dataset (2,700 in all) | +.004 [-.007, +.016] | -.001 [-.011, +.010] | +.002 [-.013, +.020] | +.005 [-.013, +.025] |
| Learning rate 2e-4 (4 times higher) | -.005 [-.019, +.007] | +.015 [+.001, +.032] | -.004 [-.017, +.017] | +.069 [+.049, +.091] |

- With 900 questions the training loss stays flat at about 0.44 (mean of the first 50 steps .417, of
  the last 50 .459). Each logged step is the mean of 8 questions and is very noisy, so use
  `train/loss_avg30` in wandb, not the raw curve.
- At 2e-4 the answers change more with the option order (flip rate .088 to .176 on BoolQ). The loss
  fell a little during training, but validation did not follow, so training loss is not a good signal here.
- Accuracy stays within noise of B0 in all three runs. More public data or a different step size did not help.
  The OpenDecisions results above show why: the public questions are not decision-shaped.

**Training on the hardest questions did not help either** (`run.py --hard`). From the first 900 train
questions of each dataset it keeps the 350 where the base model has the highest loss. Two seeds, each minus B0,
all 750 validation questions, each model with its own temperature:

| Seed | Accuracy | NLL | Flip rate | CSQA accuracy |
|---|---|---|---|---|
| 0 | +.008 [.000, +.017] | +.015 [-.005, +.037] | -.011 [-.032, +.011] | +.028 [+.004, +.052] |
| 1 | +.003 [-.011, +.016] | +.035 [+.003, +.070] | -.027 [-.049, -.007] | +.016 [-.016, +.048] |

- Seed 0 alone looked like a gain on CSQA (7 questions fixed, 1 broken). Seed 1 does not repeat it, so the
  gain is within the seed noise.
- Calibration gets worse in both seeds, mostly on BoolQ (NLL +.05, ECE +.06 and more). The hardest BoolQ
  questions probably carry noisy labels, so the model learns to be over-confident. The fitted temperature
  moves between seeds (.529 and .401).
- The CSQA flip rate falls (.260 and .240), but that is inside the range of the random-350 seeds (.248 to .312).

**About "RLCD".** Open models that use the name (Laya, eve-rlcd, von and others) do not show reinforcement
learning beating supervised cross-entropy plus temperature scaling. Laya's own ablation found no gain
from the RL term. So this repo has no RL stage.

**Limits.** On the public data: three training seeds at 350 questions, two with `--hard`, one at 900 and one at learning rate 2e-4. The bootstrap
covers the choice of questions, not the training seed, so it understates the real uncertainty.
ECE with 10 bins is biased on 250 questions, so NLL is the cleaner number. One temperature is fitted
across datasets that have 2, 4 and 5 options. The public datasets are easy for this model, so there is
little room for accuracy to rise.
