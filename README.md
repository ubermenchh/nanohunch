<p align="center"><img src="assets/banner.png" width="100%" alt="nanohunch banner"></p>

# nanohunch

Calibrated decision probabilities from a 4B language model with a LoRA adapter.

## Overview

nanohunch takes a text state and one typed question (choice, yes/no, or score) and returns a probability for each answer. The
probabilities are the softmax of the model's logits for the answer labels at the last token, so each question needs one forward pass and
no text is generated.

- **Model:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) with a LoRA adapter (rank 16), trained on 4,000
  decision questions with soft targets (the teacher's probability instead of a hard label).
- **Benchmark:** [OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions) (OD), 2,352 held-out questions. Capability is
  (Intelligence + Calibration) / 2 from the dataset's own `score.py`; its authors consider differences under 2 points not significant.
- **Result:** Capability 57.1 (mean of four seeds, standard deviation 1.0), against 53.1 for the untrained Qwen3-4B-Base.
- **Weights:** [huggingface.co/ubermenchh/nanohunch-decider-4b](https://huggingface.co/ubermenchh/nanohunch-decider-4b), in PEFT and MLX formats.

This is a completed research project. The repository records the experiments, including the ones that did not help.

## Results

Capability on OpenDecisions. The mean and standard deviation are over training seeds; "vs base" compares with the untrained Qwen3-4B-Base (53.1).

| Recipe | Seeds | Capability per seed | Mean (std) | vs base |
|---|---|---|---|---|
| Qwen3-4B-Base, untrained | n/a | 53.1 | 53.1 | 0 |
| Qwen3-4B-Instruct-2507, untrained | n/a | 35.4 | 35.4 | -17.7 |
| Base + LoRA on public questions (BoolQ, ARC, CSQA) | 1 | 51.4 | 51.4 | -1.7 |
| Instruct + decider data, hard labels, fitted temperature | 4 | 56.7, 54.1, 48.5, 52.2 | 52.9 (3.5) | -0.2 |
| Same, with the four seeds' probabilities averaged | | | 55.9 | +2.8 |
| Instruct + decider data, soft targets, fitted temperature | 4 | 54.6, 55.7, 52.0, 54.3 | 54.2 (1.6) | +1.1 |
| **Instruct + decider data, soft targets, temperature 1** | **4** | **57.5, 58.3, 56.0, 56.4** | **57.1 (1.0)** | **+4.0** |

Main findings:

1. **Decision-shaped training data matters.** Fine-tuning on public gold-label questions (BoolQ, ARC, CommonsenseQA) did not move OD.
   Fine-tuning on 4,000 typed decision questions from [decider](https://github.com/Mapika/decider) raised Intelligence by 6 to 12 points over the
   untrained Instruct model in all eight trained seeds.
2. **Soft targets make explicit calibration unnecessary.** With soft targets, the model's probabilities are already usable. Fitting a
   temperature on held-out decider questions made OD scores worse (-2.1 to -4.7 points), so the final recipe uses temperature 1. Soft targets
   also reduced the spread between seeds from 3.5 to 1.0 standard deviations.
3. **Nothing else tried helped.** More data, hard-only data, rule-generated data, a hard-heavy mix, a larger model (Qwen3-8B), a Brier loss term and
   a higher learning rate all stayed within seed noise or scored lower (see [Experiments](#experiments)).
4. **The model remains well below the leaderboard.** The OD leaderboard lists 4B fine-tunes at 67.7 (Jeff-1.0-Qwen-4B) and 77.1 (Wald-4B v2.0).

<details>
<summary>Full per-seed results (Capability, Intelligence, Calibration, accuracy by question type)</summary>

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

T = temperature. "Fitted T" is one temperature per question type, fitted on held-out decider questions; "T = 1" applies no scaling. Accuracy by
type does not depend on the temperature. The hard-label seed 1 score comes from rescaling saved predictions to per-type temperatures.
"(Modal)" marks runs trained on a Modal H100 with MLX's CUDA backend; the others were trained on an Apple Silicon Mac. A Modal-trained adapter
loads on the Mac and gives the same probabilities to within 0.04.

</details>

## Usage

The adapter is published in two formats: PEFT (`adapter_model.safetensors`, for transformers and peft) and MLX (`adapter.npz`, for this repository's code).
The first run downloads Qwen3-4B-Instruct-2507 (about 8 GB) and the adapter (132 MB).

### transformers and peft

Requires `pip install torch transformers peft`. The PEFT file was converted from the MLX file with [to_peft.py](to_peft.py) and checked against it on
40 benchmark questions: the same top answer on all 40, with probabilities within 0.022.

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

Options are written `name: description`. A yes/no question has the two options `true: ...` and `false: ...`; a score question has levels
`0: ...`, `1: ...`, lowest first. The output is `refund 0.962, replacement 0.021, info 0.017`, then `true 0.593, false 0.407`. Use the logits as they are
(temperature 1). The snippet was run on a CPU.

### MLX (this repository)

Requires an Apple Silicon Mac (tested on 24 GB; the model needs about 8 GB) or Linux with `mlx[cuda12]`. [ask.py](ask.py) accepts the typed question format directly.

```bash
git clone https://github.com/ubermenchh/nanohunch && cd nanohunch
uv sync
uv run python ask.py        # runs three example questions
```

**Ask your own question.** [ask.py](ask.py) loads the adapter once and exposes `ask(state, question)`. Open it, edit the examples at the bottom, or
`from ask import ask` in your own script (importing it loads the model). A question is a dict with one of three types (`noul` is the name for yes/no questions):

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

- `state` can be text, or a dict or list (it is written out as JSON).
- Up to 8 options per question. Probabilities refer to the options in the order given. Reordering the options can change the answer slightly
  (the flip rate is measured in [docs/public-datasets.md](docs/public-datasets.md)); average over several orders for a steadier estimate.
- The model reads the answer once, at the last token: a single forward pass, with no text output or explanation.
- The temperatures in `config.json` are all 1 (see [Soft targets and temperature](#soft-targets-and-temperature)).
- To compare with the plain Instruct model, comment out the `model.load_weights(...)` line in `ask.py`.
- To score a file of questions as the benchmark does, use `od.py` (see [Reproduce](#reproduce)).

## How it works

**Readout.** The prompt is the state, then `Question: ...`, then the options labelled `A.` to `H.` (at most 8), then `Answer:`. The logits of the label
tokens at the last position are taken from the output head and passed through a softmax over the options. Yes/no questions have two options
(true, false); score questions have one option per level. Option order is shuffled during training and the gold label is remapped.

**Training.** LoRA on all seven projections (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`) of all 36 layers, rank 16, alpha 32, AdamW, learning
rate 1e-4 with 10% warmup and cosine decay, 32 questions per optimizer step, a single pass of 150 steps. The data is 4,000 teacher questions plus 810
public replay questions (270 each from BoolQ, ARC-Challenge and CommonsenseQA). The loss is cross-entropy over the answer labels. With soft
targets, the gold option receives the teacher's probability `p` and the other options share `1 - p` equally.

**Teacher data.** decider's `custom_questions.jsonl` (Apache-2.0, pinned commit `91b6fa9`) has 23,392 typed questions (8,875 yes/no, 8,505 choice,
6,012 score), each with a teacher probability `teacher_p`. Questions with `teacher_p >= 0.5` are kept (22,514); 2,152 of them are "hard" (`teacher_p < 0.8`).

**Evaluation.** [od.py](od.py) scores a model or adapter with one pass per question and the options in their given order, as on the leaderboard.
Predictions are scored with the dataset's `score.py`. OD has 1,178 choice, 586 yes/no and 588 score questions. OD is never used for training or for
fitting a temperature.

**Implementation.** The code is a small set of flat Python files on [MLX](https://github.com/ml-explore/mlx), Apple's array framework. LoRA, the
loss and the training loop are implemented directly rather than through a trainer library.

| File | What it does |
|---|---|
| `model.py` | Load the model, LoRA layers, read the answer-label logits at the last token |
| `data.py` | Question to prompt, option shuffling with the gold remap, public datasets, decider loader |
| `train.py` | Loss over the answer labels (hard or soft target) and the training loop |
| `eval.py` | Pooling over option orders, accuracy, flip rate, ECE, temperature fit |
| `run.py` | Phase 1: experiments on public datasets (see `docs/public-datasets.md`) |
| `run_decider.py` | Phase 2: train on decider's teacher questions (`--soft`, `--hard`, `--rules`) plus public replay |
| `ask.py` | Ask the adapter from Hugging Face a typed question (a runnable example) |
| `to_peft.py` | Convert an adapter from MLX `.npz` to PEFT format |
| `od.py` | Score a model or adapter on OpenDecisions (`--temp 1` to ignore the fitted temperatures) |
| `refit_temp.py`, `retemp.py` | Fit one temperature per question type on hard held-out questions; rescale saved predictions offline |
| `make_rules.py` | Generate rule-conditioned questions with decider's generator |
| `analyze.py` | Paired-bootstrap intervals for phase 1 |

## Experiments

Each row changes one thing from the baseline named in the third column. Differences under about 3 points are within the seed spread of the
hard-label recipe (standard deviation 3.5), so only the first and third rows are clear results.

| Change | Result (Capability) | Baseline for comparison | Verdict |
|---|---|---|---|
| Train on decision-shaped teacher questions instead of public ones | Intelligence 31.0 to 36.7-43.2 across the 8 seeds; accuracy +2 to +12 points by question type | Untrained Instruct | **Worked.** Solid in all 8 trained seeds |
| Soft targets (`--soft`) | 54.2 (std 1.6) | 52.9 (std 3.5) hard labels, same temperature fit | Small gain, and the seed spread halves |
| Soft targets, no temperature scaling | 57.1 (std 1.0) | 54.2 soft targets with fitted T | **Worked.** +2.9, and +4.2 over hard labels |
| One temperature per question type, instead of one | +1.9 (seed 0: 54.8 to 56.7), +1.2 (seed 1: 52.9 to 54.1) | One temperature | Small. Chosen after seeing an OD score, so the gain is optimistic |
| Fit the temperature on random held-out decider questions | 50.1 (seed 0) | 54.8, fitted on hard held-out questions | Hurts: those questions are easy (median teacher probability .95) |
| Fit the temperature on 1,000 instead of 300 hard questions | 56.3, 54.0, 48.3, 52.6 (hard labels) | 56.7, 54.1, 48.5, 52.2 | No change (within 0.5). The temperature fit is not the source of the seed spread |
| 12,000 teacher questions instead of 4,000 | 55.0 | 56.7 (seed 0) | No gain. Accuracy flat, calibration lower, training loss 0.025 (over-fit to easy questions) |
| Only the 1,850 hard teacher questions (teacher probability under .8) | 51.4 | 54.8 (one temperature, 4,000 q) | Worse. Calibration lowest of all runs; fitted T .90 does not transfer |
| 2,000 teacher + 2,000 rule-generated questions | 54.2 | 56.7 (seed 0) | No measurable gain (within seed spread), with half the teacher data |
| Hard-heavy mix: 1,500 hard + 2,500 easy, soft targets | 49.5, 50.6 | 54.6, 55.7 (soft, fitted T, same seeds) | Worse. Intelligence unchanged; hard-tier accuracy 49.6% against 50.8% |
| Qwen3-8B instead of the 4B, soft targets, T = 1 | 55.1, 57.0 | 57.5, 58.3 (4B, same seeds) | No gain. Intelligence about 42 for both |
| Cross-entropy + 0.5 x Brier loss (public data) | NLL -.002 [-.003, -.000] | Cross-entropy | No real gain. Reverted (commit `33ad703`) |
| 900 questions per dataset, learning rate 2e-4 (public data) | Accuracy within noise of the untrained model | 350 questions, lr 5e-5 | No gain. See [docs/public-datasets.md](docs/public-datasets.md) |
| Hardest public questions only (`run.py --hard`) | Accuracy within noise; calibration worse | Random 350 | No gain. See [docs/public-datasets.md](docs/public-datasets.md) |

### Soft targets and temperature

Each decider question carries `teacher_p`, the teacher's probability for the gold answer. A temperature fitted on held-out decider questions came out
below 1 for soft-target models (0.65 to 1.1, mostly under 1). On the easy decider questions the soft-target model is under-confident, so the fit
sharpens it, and on the harder OD questions that makes it over-confident. Removing the temperature (`od.py --temp 1`) raised Capability by
2.1 to 4.7 points in all six soft-target runs (four 4B seeds, two 8B seeds). The effect was found on seeds 0 and 1 and then confirmed on seeds 2 and 3
(52.0 to 56.0 and 54.3 to 56.4). Hard-label models are over-confident and still need a fitted temperature, so the comparison is soft targets without a
temperature against hard labels with one.

### Where the remaining errors are

Each OD question has a difficulty tier, a family, and a distractor (a plausible wrong option). The table uses the averaged predictions of the four
soft-target 4B seeds with their fitted temperatures (the analysis predates the temperature finding, so confidences are slightly high).

| Tier | Questions | Accuracy | Mean confidence | Picks the distractor |
|---|---|---|---|---|
| Easy | 443 | 87.1% | 0.87 | 2.7% |
| Standard | 407 | 74.7% | 0.79 | 14.5% |
| Judge | 581 | 71.9% | 0.75 | 12.7% |
| **Hard** | **921** | **49.6%** | **0.73** | **30.5%** |

- Confidence matches accuracy on easy questions. On the hard tier the model is right about half the time at 73% confidence and picks the distractor in
  30% of cases.
- Lowest accuracy by family: `usecase` 42%, `fairness` 52%, `jaggedness` 52%, `perturb_pair` 57%, `multilingual` 64%. Highest: `llm_router` 77% and
  `negation_pair_gen` 99%.
- Lowest accuracy by use case: insurance claims, financial crime, risk assessment and LLM guardrails (about 50%). Several non-English languages
  score below English (67%), for example Spanish 56% and Arabic 52%, with only 8 to 46 questions per language.
- decider's training data covers these domains (by keyword match, about 22% of its 3,043 states mention finance or fraud, 9% insurance, 10% legal, 42%
  support) but mostly with easy questions: only 10% of its questions are hard. Adding more hard questions did not help.

Phase 1 of the project, experiments on public datasets, is described in [docs/public-datasets.md](docs/public-datasets.md).

## Implementation notes

- **LoRA dtype.** Float32 LoRA weights promoted the activations to float32. All-layer LoRA used 24.8 GB and 45 s per step; casting the LoRA output
  back to the input dtype reduced this to 11.6 GB and 2.1 s per step.
- **Output head.** `answer_logits` first read the embedding matrix as the output head, which is correct for Qwen3-4B (tied embeddings) but not for
  Qwen3-8B, which has a separate `lm_head`. The untrained 8B scored Capability 6.5 (choice accuracy at chance, 26%) until the function read `lm_head`
  when present. The 4B results are not affected.
- **Seed variance.** Seed 0 of the hard-label recipe scored 56.7, but the four seeds range from 48.5 to 56.7. The first published adapter came from
  that best seed and was replaced after the other seeds were run.
- **Modal.** `modal run --detach` with a `.map` over jobs cancelled all jobs but the last when the local process disconnected. `modal deploy` with
  `Function.spawn` does not have this problem.

## Limitations

- **Few seeds.** The soft-target gain (+4.2 over hard labels) is about 2.3 standard errors of the difference of the means (standard deviations 3.5 and
  1.0, four seeds each), and those deviations are estimated from four runs. The 8B and hard-heavy results have two seeds each.
- **Choices made after seeing OD scores.** Per-type temperatures and temperature 1 were chosen after looking at OD results. Seeds 2 and 3 confirmed
  the temperature 1 choice, but the recipe has been tuned against this benchmark, so 57.1 is likely optimistic.
- **Mixed hardware.** Some seeds were trained on a Mac and some on CUDA. Results differ by about 1 point from numerics, and the soft-target comparison
  mixes both.
- **Calibration.** Calibration is 72 to 75, below the untrained base model's 80. The base model is inaccurate but uniformly unsure, which scores well
  on calibration.
- **Scope.** One benchmark and one prompt format, one pass per question with the options in the given order. Other formats and longer states were not
  tested, and the final adapter was not checked for regressions on the public datasets.

## Reproduce

```bash
uv sync
uv run python run_decider.py --soft --seed 0                           # trains; writes adapter_decider_soft_n4000.npz and .json
uv run python od.py --adapter adapter_decider_soft_n4000.npz --temp 1  # writes runs/od_adapter_decider_soft_n4000.jsonl
python3 -I score.py runs/od_adapter_decider_soft_n4000.jsonl           # score.py is in the OpenDecisions dataset repository
```

Add `--small` to `run_decider.py` for a quick check. Runs log to wandb unless `--no-wandb` is given. The Modal runs used `mlx[cuda12]==0.32.4` on an
`nvidia/cuda:12.9.1-devel-ubuntu22.04` image with one H100 per run (about 35 minutes for 4B training, refit and scoring; 10 to 15 minutes of training for 8B).
The launch scripts are not included in the repository.

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
- Karpathy. **nanoGPT**, the model for this repo's small, flat code layout: https://github.com/karpathy/nanoGPT
