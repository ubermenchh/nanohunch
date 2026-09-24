# ADR 0003: Prompt and serialization format nanohunch-fmt-v1

**Status:** accepted (tokenizer checks run in Phase 0)
**Date:** 2026-09-23
**Door:** one-way (expensive to reverse)
**Deciders:** @umang (author), sd-architect

## Context

The format is baked into every LoRA adapter and into every teacher soft label
read from logits. Changing it after the first paid run invalidates checkpoints
and logit-read labels. It also decides whether state sharing works at all: the
11x W1 saving (DERIVED, `REQ:169-171`) requires the state to be a byte-identical
prefix of every question. pngwn's long-input regression came from training at
384 tokens (`RN:140-142`), so the format must not bound the state length.

## Decision

We will render every request as raw text with no chat template:

```
You will answer questions about the STATE. Answer with the label of exactly one option.

### STATE
{state}
### END STATE

### QUESTION
{question text}
{option lines}
Answer:
```

Rules:

1. **Prefix** = preamble + state block through the blank line after
   `### END STATE`. Everything after is the **branch**.
2. **Split tokenization:** prefix and branch are tokenized separately and their
   ids concatenated, in training, teacher labelling, oracle and serving. The
   joined text is never re-tokenized.
3. **Option lines:** Choice `A. {opt}` to `Z. {opt}`; Score `{value}: {label}`
   with integer values 0..9 ascending; Noul the single line `(yes/no)`.
4. **Readout** at the last token of `Answer:`; label tokens are space-prefixed
   (` A`, ` 7`, ` yes`) and must each encode to exactly one token id, checked
   when the formatter loads a tokenizer (`LabelNotSingleToken` otherwise).
5. Caps: 26 options, 10 levels. State length bounded only by `max_context`.
6. The string `nanohunch-fmt-v1` is recorded in every training row, teacher label,
   calibration file, model card and `/v1/info` response.

## Alternatives considered

| Option | Why it lost |
|---|---|
| Question before state | Breaks prefix sharing: every question would need its own state encode (16,896 vs 1,536 tokens at W1, `REQ:169-170`) |
| Chat template of an instruct model | Base models are trained on raw text; the template adds tokens and varies by family, making the Qwen3/MiniCPM bake-off unequal |
| Trailing space after `Answer:` with unprefixed labels | Common tokenizers attach the space to the next word, so a trailing space pushes the model into rare token splits |
| Numeric labels `1..N` for Choice | Qwen tokenizers split digits, so labels above 9 are multi-token; letters give 26 single-token labels |
| Letters A/B for Noul | Adds an order to a question that has none; the words carry meaning for the zero-shot model |

## Consequences

**We gain:** state-once branching by construction; identical inputs for
teacher, student, oracle and server; B0 and fine-tune measured on the exact same
text.

**We accept:** 26 options and 10 levels maximum in v1; Score values restricted
to 0..9; a fixed yes/no order in the Noul prompt, whose constant bias the
calibrator may absorb with a bias term (`architecture.md` section 6.4).

**We are now locked into:** all adapters and logit-read teacher labels. A v2
format means relabelling logit-read labels (GPU time) and retraining.

**Revisit if:** a real eval workflow needs more than 26 options in > 1% of
Choice items, or Score scales above 10 levels, or Phase 0 finds a label token
that is not a single token in the chosen tokenizer.

## Verification

Unit tests: every label encodes to one id for each candidate tokenizer; for 100
fixtures, `prefix_ids + branch_ids` is identical between the training loader
and the server path; branched vs re-encoded probabilities agree within 1e-3 in
bf16 (`REQ:260`).
