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

Status: early. On public gold data the base model is already at .82 to .90 accuracy, and
LoRA training did not change accuracy measurably in two runs. It did improve calibration
and the order sensitivity of answers. See `run.py` for the exact setup.
