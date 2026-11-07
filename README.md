# nanohunch

A small System One model. Qwen3-4B-Base reads a text state and one typed question
(Choice or yes/no) and returns a calibrated probability for each answer, not text.

Everything is written by hand in MLX on an Apple Silicon Mac: label-token readout, LoRA,
restricted softmax cross-entropy, option-order shuffling, temperature scaling.

| File | What it does |
|---|---|
| `model.py` | Load the model, LoRA layers, read the answer-label logits at the last token |
| `data.py` | Question to prompt, option shuffling with the gold remap, public datasets (BoolQ, ARC, CSQA) |
| `train.py` | Loss over the answer labels and the training loop |
| `eval.py` | Pooling over option orders, accuracy, flip rate, ECE, temperature fit |
| `run.py` | The whole experiment in `# %%` cells: base model, train, calibrate, report |

Run it with `uv run python run.py` (add `--small` for a quick check, `--seed S` and `--n N` to change the seed and
the number of training questions per dataset). It logs to wandb.

## Results

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
- Accuracy stays within noise of B0 in all three runs. The limit is the labels and the task, not the amount
  of training or the step size.

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

**Limits.** Three training seeds at 350 questions, two with `--hard`, one at 900 and one at learning rate 2e-4. The bootstrap
covers the choice of questions, not the training seed, so it understates the real uncertainty.
ECE with 10 bins is biased on 250 questions, so NLL is the cleaner number. One temperature is fitted
across datasets that have 2, 4 and 5 options. The public datasets are easy for this model, so there is
little room for accuracy to rise.
