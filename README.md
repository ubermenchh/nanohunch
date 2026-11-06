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

Run it with `uv run python run.py` (add `--small` for a quick check). It logs to wandb.

## Results

Setup: Qwen3-4B-Base, LoRA (rank 8, alpha 160, 8 layers, lr 5e-5), trained on 1,050 public gold questions
(350 each from BoolQ, ARC-Challenge and CommonsenseQA train splits) with the option order shuffled each time.
The temperature is fitted on 600 other train-split questions (200 each), never on training or validation
questions. Scores are on 750 validation questions (250 each). One seed (0). The same seed gives the same result,
checked twice. Intervals are 95%, from a paired bootstrap that resamples the questions (`analyze.py`).

**The base model is already strong on this data.** B0 accuracy is .892 (BoolQ), .896 (ARC), .820 (CSQA).

**Trained minus B0, all 750 questions** (a negative number is better for NLL, ECE and flip):

| | Accuracy | NLL | ECE | Flip rate |
|---|---|---|---|---|
| Raw probabilities | 0.000 [-.009, +.009] | -.082 [-.095, -.069] | -.086 [-.093, -.067] | -.025 [-.044, -.007] |
| Each model with its own temperature | +.001 [-.008, +.011] | -.014 [-.024, -.005] | -.011 [-.024, +.008] | -.025 [-.044, -.007] |

- Accuracy does not change, to within about one point.
- Training makes the raw probabilities much better calibrated. Most of that gain is also available
  from temperature scaling alone (B0 ECE falls from .065 / .149 / .180 to .019 / .054 / .045).
- After scaling, a small gain in loss remains. Answers change less when the options are reordered.
  The drop is clearest on CSQA (flip rate .316 to .248, interval [-.112, -.028]).
- B0 is under-confident (temperature .554), not over-confident. Pooling the probabilities over three
  option orders softens them.

**Tried and dropped: cross-entropy plus 0.5 x Brier loss** (commit `33ad703`, reverted after it).
Brier minus cross-entropy, both with their own temperature, all 750 questions: accuracy 0.000,
NLL -.002 [-.003, -.000], ECE -.002 [-.011, +.010], flip -.003 [-.008, +.003]. The NLL difference is
smaller than the spread between two cross-entropy runs with different starting weights (up to .007),
so it is not a real gain. Plain cross-entropy with a held-out temperature is enough.

**About "RLCD".** Open models that use the name (Laya, eve-rlcd, von and others) do not show reinforcement
learning beating supervised cross-entropy plus temperature scaling. Laya's own ablation found no gain
from the RL term. So this repo has no RL stage.

**Limits.** One training seed. The bootstrap covers the choice of questions, not the training seed.
ECE with 10 bins is biased on 250 questions, so NLL is the cleaner number. One temperature is fitted
across datasets that have 2, 4 and 5 options. The public datasets are easy for this model, so there is
little room for accuracy to rise.
