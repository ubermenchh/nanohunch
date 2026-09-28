# ADR 0003: Prompt and serialization format nanohunch-fmt-v1

**Status:** accepted, with Amendment 1 (Score labels: option (b), decided 2026-09-24)
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
   (` A`, ` yes`) and must each encode to exactly one token id, checked
   when the formatter loads a tokenizer (`LabelNotSingleToken` otherwise).
   **Score, per Amendment 1:** the branch ids end with the tokenizer's space id
   appended after `Answer:`, and the readout reads the bare-digit rows `0`..`9`
   at that appended position.
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
**fp32, unbatched, on the CPU backend** (P0-4), and branch isolation diff is exactly 0 (`risks.md` R8;
bf16 differs by about 1.3e-2 by rounding, so it is not the correctness check).

## Amendment 1 (decided 2026-09-24: option (b))

**Measured 2026-09-24:** in the MiniCPM5-2B-Base tokenizer, ` A`..` Z`, ` yes`
and ` no` are single ids, but ` 0`..` 9` each encode to **two** ids (a space
token, 242, then the digit). Rule 4 as written therefore fails for every Score
question on the MVP model. Choose one scheme before the Phase 2 freeze:

| Option | Score readout | Gains | Costs |
|---|---|---|---|
| (a) Letters for Score | option lines `A. {label}` … in ascending value order; labels ` A`..` J` map to values 0..9 | one readout path for Choice and Score; already known single-token | loses the digit prior the zero-shot base has for ratings; `Answer.expected` maps letters back to values |
| (b) Space id, then digit | branch ids end with `Answer:` followed by the space id **appended as an id** (rule 2 still holds: never re-tokenized text); read the bare-digit rows at that last position | keeps digit meaning and the ADR's option-line text | one extra branch token for Score; must confirm `encode("Answer: 7") == ctx + [space_id, id("7")]` in both candidate tokenizers, and that the space id is context-independent |

**Decision (2026-09-24, @umang): option (b).** P0-1 output
(`reports/phase0.md`): Choice and Noul single-token in both tokenizers; ` 0`..` 9`
two ids in both (MiniCPM5 `[242, d]`, Qwen3 `[220, d]`); for option (b),
`encode("Answer: ")` is `encode("Answer:")` plus exactly one space id (242
MiniCPM5, 220 Qwen3) and `encode("Answer: " + d) == encode("Answer:") +
[space_id, id(d)]` for all ten digits in both. So (b) is the tokenizer's own
split of `"Answer: 7"`, and the base model keeps its digit prior for ratings.

How it is applied:
- Score branch text is unchanged (it still ends in `Answer:`). The branch ids are
  `encode(branch_text) + [space_id]`, where `space_id = encode("Answer: ")[-1]`
  after checking that it is exactly one extra id. The id is appended, never the
  text re-tokenized, so rule 2 holds.
- `label_vocab(tok, "score", n)` returns the ids of the bare digits `"0"`..`"n-1"`,
  each checked to be one id.
- Choice and Noul are unchanged.
- Teachers are unaffected: they see the same text, and candidate matching strips
  the leading space (`"7"` and `" 7"` both count).
- Checked by `uv run python -m bench.check_labels` (default scheme `space_digit`).
