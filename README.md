# nanohunch

The simplest, smallest repository for training a System One model: a small open LLM that reads a
state, answers many typed questions about it at once, and returns calibrated probabilities instead
of text.

> **Status: planning done, build starting.** Nothing below is measured yet. The numbers are
> targets until the plan's phases measure them. See [the plan](docs/design/2026-09-23-nanohunch/plan.md).

- **Typed answers, never text.** Choice, Score and yes/no questions; the answer is always one of
  the options you supplied, with a probability for each.
- **One state, many questions.** The state is encoded once; each question reuses the KV cache.
- **Aims to be calibrated.** Target: confidence 0.8 means right about 80% of the time, measured
  on a held-out split (ECE after temperature scaling at or below 0.05).
- **Minimal.** Six core files, under 1,000 lines, readable end to end (enforced by `tools/loc.py`).
- **Trains on a laptop.** LoRA on MiniCPM5-2B-Base in MLX on one Apple Silicon Mac.

## How it works (planned)

```
state ──► [ encode once: KV cache ] ──┬──► question 1 ─► logits of " A" " B" " C" ─► softmax / T ─► [0.71, 0.21, 0.08]
                                      ├──► question 2 ─► logits of " yes" " no"   ─► softmax / T ─► p_yes = 0.93
                                      └──► question 3 ─► logits of " 0" .. " 4"   ─► softmax / T ─► E[score] = 2.6
```

1. Render the state first, then each question as a short branch ending in `Answer:`.
2. Encode the state once; run each branch on the cached state.
3. Read only the LM-head rows of the answer labels and softmax over them. The output is valid by
   construction.
4. Divide by a temperature fitted on a calibration split.
5. Train a LoRA adapter on soft labels from two open teacher models, with option order shuffled
   every epoch so position is never a shortcut.

## Core layout (planned)

| File | Budget | Contents |
|---|---|---|
| `fmt.py` | 130 | question types, prompt format, label vocab |
| `engine.py` | 200 | chunked prefill, cache branching, label readout |
| `calibrate.py` | 170 | temperature scaling, accuracy, ECE, NLL, Brier, bootstrap |
| `train.py` | 220 | LoRA, restricted soft cross-entropy, training loop |
| `dataset.py` | 150 | rows keyed by option id, splits, teacher-label pooling |
| `evaluate.py` | 130 | eval harness |

## Measured against

Results will be reported in one harness against the model's own untrained base, pngwn's open
replication (order flips, long inputs), SemIf-OpenJev (frozen Qwen3.5-4B), and decider-2b (a
trained 2B model). JevBench public items are reported as self-run, not as an official score.

## Docs

- [Design doc](docs/design/2026-09-23-nanohunch/README.md): what is being built and why.
- [Plan](docs/design/2026-09-23-nanohunch/plan.md): eight phases, about 160 hours, each with a
  gate, a rollback and kill criteria.
- [Risks](docs/design/2026-09-23-nanohunch/risks.md): what could go wrong and what was decided.

Inspired by TypeSafe's Jev and Karpathy's nanoGPT. Not affiliated with TypeSafe.

## License

MIT
