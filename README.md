# nanohunch

A small System One model. A Qwen3-4B model reads a text state and one typed question
(choice, yes/no or score) and returns a calibrated probability for each answer, not text.

Everything is written by hand in MLX on an Apple Silicon Mac: label-token readout, LoRA,
restricted softmax cross-entropy, option-order shuffling, temperature scaling.

| File | What it does |
|---|---|
| `model.py` | Load the model, LoRA layers, read the answer-label logits at the last token |
| `data.py` | Question to prompt, option shuffling with the gold remap, public datasets (BoolQ, ARC, CSQA) |
| `train.py` | Loss over the answer labels and the training loop |
| `eval.py` | Pooling over option orders, accuracy, flip rate, ECE, temperature fit |
| `run.py` | The public-data experiment in `# %%` cells: base model, train, calibrate, report |
| `run_decider.py` | Train on decision-shaped questions (decider's teacher data, hard or soft targets) plus public replay |
| `od.py` | Score a model or adapter on OpenDecisions, one pass per question |
| `refit_temp.py`, `retemp.py` | Fit one temperature per question type on harder held-out questions; rescale saved predictions |

Run the public-data experiment with `uv run python run.py` (add `--small` for a quick check, `--seed S` and `--n N` to change the seed and
the number of training questions per dataset). It logs to wandb.

The adapter (4,000 decider questions, soft targets, no temperature scaling, seed 0: 57.5, the second best of four seeds, mean 57.1) is at
[huggingface.co/ubermenchh/nanohunch-decider-4b](https://huggingface.co/ubermenchh/nanohunch-decider-4b).

## Results on OpenDecisions

[OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions) holds 2,352 held-out decision questions
(choice, yes/no, score). It is never trained on, and no temperature is fitted on it. Capability is the dataset's own
`score.py`: (Intelligence + Calibration) / 2. Its authors say differences under 2 points are not significant.

| Model | Capability | Intelligence | Calibration | choice | yes/no | score |
|---|---|---|---|---|---|---|
| Qwen3-4B-Base, no training | 53.1 | 26.3 | 80.0 | 55.4% | 68.9% | 40.8% |
| Base + LoRA on BoolQ/ARC/CSQA (n900) | 51.4 | 24.4 | 78.4 | 55.5% | 67.6% | 40.3% |
| Qwen3-4B-Instruct-2507, no training | 35.4 | 31.0 | 39.8 | 62.4% | 69.6% | 45.7% |
| Instruct + decider data, 4,000 q, seed 0 | 56.7 | 42.8 | 70.7 | 68.3% | 74.4% | 57.1% |
| Instruct + decider data, 4,000 q, seed 1 | 54.1 | 40.3 | 67.8 | 67.3% | 73.0% | 53.9% |
| Instruct + decider data, 4,000 q, seed 2 | 48.5 | 36.7 | 60.3 | 66.6% | 71.8% | 50.3% |
| Instruct + decider data, 4,000 q, seed 3 | 52.2 | 39.6 | 64.8 | 67.1% | 73.7% | 51.0% |
| Instruct + decider data, 4,000 q, soft targets, seed 0 (Modal) | 54.6 | 40.5 | 68.8 | 68.3% | 73.2% | 54.1% |
| Instruct + decider data, 4,000 q, soft targets, seed 1 (Modal) | 55.7 | 43.2 | 68.1 | 68.9% | 75.1% | 55.3% |
| Instruct + decider data, 4,000 q, soft targets, seed 2 | 52.0 | 39.7 | 64.3 | 66.8% | 73.4% | 56.0% |
| Instruct + decider data, 4,000 q, soft targets, seed 3 | 54.3 | 40.7 | 67.9 | 66.6% | 74.2% | 53.1% |
| Instruct + decider data, 4,000 q, soft targets, **temperature 1**, seed 0 (Modal) | 57.5 | 40.3 | 74.6 | 68.3% | 73.2% | 54.1% |
| Instruct + decider data, 4,000 q, soft targets, **temperature 1**, seed 1 (Modal) | 58.3 | 43.0 | 73.6 | 68.9% | 75.1% | 55.3% |
| Instruct + decider data, 4,000 q, soft targets, **temperature 1**, seed 2 | 56.0 | 39.3 | 72.7 | 66.8% | 73.4% | 56.0% |
| Instruct + decider data, 4,000 q, soft targets, **temperature 1**, seed 3 | 56.4 | 40.6 | 72.3 | 66.6% | 74.2% | 53.1% |
| Qwen3-8B + decider data, 4,000 q, soft targets, temperature 1, seed 0 (Modal) | 55.1 | 41.9 | 68.3 | 68.9% | 72.7% | 57.8% |
| Qwen3-8B + decider data, 4,000 q, soft targets, temperature 1, seed 1 (Modal) | 57.0 | 42.1 | 71.9 | 68.2% | 74.2% | 57.3% |
| Instruct + decider data, 12,000 q, seed 0 | 55.0 | 43.2 | 66.8 | 68.9% | 75.1% | 54.8% |
| Instruct + only the hard decider q, 1,850 q, seed 0 (one temperature) | 51.4 | 40.6 | 62.2 | 66.0% | 75.1% | 54.1% |
| Instruct + 2,000 teacher q + 2,000 rule q, seed 0 | 54.2 | 40.1 | 68.3 | 67.8% | 72.0% | 57.8% |

All decider rows use one temperature per question type unless marked temperature 1, except the hard-question row (one temperature). Setup: Instruct start, LoRA (rank 16, alpha 32) on all seven
projections of all 36 layers, learning rate 1e-4 with warmup and cosine decay, 32 questions per step, one pass over the
decider teacher questions (Apache-2.0, pinned commit) plus 810 public questions as replay.

- **The training data was the problem, not the method.** Training on public gold questions never moved
  OpenDecisions accuracy (all three adapters within 1.3 points of the base model). Training on decision-shaped
  questions raises accuracy by 3 to 11 points over untrained Instruct, in all four seeds.
- **Untrained Instruct is better than Base on accuracy but badly overconfident** (Calibration 39.8), so its
  Capability is lower than Base. Temperature fitted on decider questions fixes most of that.
- **The temperature must come from harder questions.** Fitted on random held-out decider questions (median teacher
  probability .95), the temperature is below 1 (.75) and seed 0 scores Capability 50.1. Fitted on held-out
  questions with teacher probability under .8, it is above 1 (1.11) and the same predictions score 54.8
  (seed 1: 52.9, only scored this way). One temperature per question type adds about 1.5 points more in both
  seeds. I chose this recipe after seeing the first OpenDecisions score, so treat the gain as optimistic.
- **Three times more data did not help.** 12,000 questions give 55.0 against 56.7 and 54.1 at 4,000. Accuracy
  is flat, calibration is a little worse, and the final training loss is 0.025, which looks like over-fitting to
  the easy questions.
- **Training only on the hard questions did not help** (`run_decider.py --hard`: the 1,850 decider questions where
  the teacher's probability for the gold answer is under .8, one temperature fitted on 300 others like them).
  Capability is 51.4 against 54.8 for the matching 4,000-question run (one temperature), and calibration is the
  lowest of the decider runs. The fitted temperature (.90) is far below the 1.1 to 1.7 the other runs needed, so the
  held-out hard decider questions do not look like OpenDecisions. These questions may also have noisier labels
  (the teacher was unsure), as with the hardest BoolQ questions, so this does not show that harder data cannot help.
- **Rule-based questions added nothing measurable** (`make_rules.py` builds decider's programmatic rule-reading
  questions: a record, a rule, the answer computed; 0 of 20,000 share a 13-word run with OpenDecisions). Half the
  4,000 questions from this set: Capability 54.2 against 56.7 for the teacher-only run, yes/no accuracy 2.4 points
  lower, score questions 0.7 higher. That sits inside the 8-point spread between the four teacher-only seeds, and
  half as many teacher questions were used, so it does not separate "no gain" from "a small loss".
- **Four seeds of the same recipe score 56.7, 54.1, 48.5 and 52.2** (mean 52.9, range 8.2). Accuracy hardly moves between
  seeds (choice 66.6 to 68.3%); Calibration does (60.3 to 70.7). Seed 0's 56.7 was a lucky draw, and no recipe difference
  above (12k questions, hard-only, rules) is larger than this spread. The mean is level with the untrained base model (53.1).
  Only the accuracy gain over untrained Instruct (Intelligence +9 to +12) is solid. Seeds 0, 2, 3 use per-type temperatures
  fitted on 300 hard held-out questions per type; seed 1 was rescaled offline.
- **Soft targets train toward the teacher's own probability** (`run_decider.py --soft`: `teacher_p` on the gold option, the rest
  spread evenly) instead of a hard label. With a fitted temperature, four seeds score 54.6, 55.7, 52.0 and 54.3 (mean 54.2,
  std 1.6) against 56.3, 54.0, 48.3 and 52.6 for hard labels (mean 52.8, std 3.4): a small gain, but the temperature fit hides
  most of the effect (next bullet). Seeds 0 and 1 were trained on Modal H100s with MLX's CUDA backend and seeds 2 and 3 on the
  Mac. Averaging the predictions of the four hard-label seeds scored 55.9.
- **With soft targets, do not fit a temperature.** The temperature fitted on held-out decider questions came out
  below 1 for soft-target models (they are under-confident on easy decider questions) and made them over-confident on
  OpenDecisions. Leaving the probabilities as trained (`od.py --temp 1`) scores 57.5, 58.3, 56.0 and 56.4 on the four 4B seeds
  (mean 57.1, std 1.0, Calibration 72 to 75), against 52.8 (std 3.4) for hard labels with fitted temperatures. I found this on
  seeds 0 and 1 and then checked it on seeds 2 and 3, which it also improved (52.0 to 56.0, 54.3 to 56.4). Hard-label models
  still need the temperature, so the comparison is soft targets without one against hard labels with one.
- **Qwen3-8B is no better than the 4B.** The same recipe on `Qwen/Qwen3-8B` (seeds 0 and 1, temperature 1) scores 55.1 and
  57.0 against 57.5 and 58.3 for the 4B; Intelligence is the same (about 42). Model size is not what limits this.
  (`answer_logits` first read the embedding matrix as the output head, which is right for the 4B but not for the 8B, whose
  head is separate; the 8B scored at chance until that was fixed.)
- **Still behind on calibration:** 72 to 75 for the best recipe against 80 for the base model, which is not trained at all and
  is simply unsure of everything. Capability is about 4 points above the base model (soft targets, temperature 1).

## Results on the public datasets

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
