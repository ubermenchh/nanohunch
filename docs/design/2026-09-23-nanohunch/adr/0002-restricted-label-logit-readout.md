# ADR 0002: Read decisions as restricted label-token logits from the frozen LM head

**Status:** accepted (order robustness confirmed or refuted by Phase 0 M2)
**Date:** 2026-09-23
**Door:** one-way (expensive to reverse)
**Deciders:** @umang (author), sd-architect

## Context

Two readouts have open evidence. A scalar cross-encoder head per (state,
question, option) scored 0.807 accuracy at 4B; a causal letter scorer with a
frozen LM head scored 0.752 at 0.6B (`RN:113-116`, MEASURED by pngwn). On
state-derivable decisions all arms were within 0.006; the gap came from world
knowledge, which tracked parameter count (`RN:121-123`). The letter scorer is
order sensitive (37.5% gold-rank flips under reversal, `RN:124`); the scalar
head is order invariant but loses gold mass at large set sizes (0.49 vs 0.61 at
26 candidates, `RN:126`) and costs one sequence per option in training, about
3x the tokens of the letter scorer for our type mix (DERIVED,
`architecture.md` section 2.3), against a 200 USD budget.

## Decision

We will read every decision from the frozen LM head: at the last token of each
question branch, compute logits only for the rows of the allowed label tokens
(` A`..` Z` for Choice, ` 0`..` 9` for Score, ` yes`/` no` for Noul) and softmax
over them. Training uses LoRA with the head frozen and cross-entropy against
teacher soft labels restricted to those rows, with random option permutation
per example. The same `label_logits` function serves training and inference.

## Alternatives considered

| Option | Why it lost |
|---|---|
| Scalar cross-encoder head (pngwn arm A) | About 3x training tokens without shared-prefix packing; new randomly initialized head means no zero-shot behaviour and no clean "fine-tune minus base" delta; set-size degradation (0.49 at 26, `RN:126`); its accuracy edge in pngwn is attributed to 4B vs 0.6B params, not the head (`RN:121-123`) |
| New classification head over options (fixed max N) | Ties the model to a max option count and loses the pretrained meaning of label tokens; not order invariant either |
| Full-vocab next-token readout with generation | Generation is out of scope (`REQ:91`); full-vocab logits OOM'd at 20.7 GiB in training (`RN:109-110`) |

## Consequences

**We gain:** one forward per question, 26 options read from one logits row for
free (`RN:130`); the untouched base is a meaningful zero-shot model
(Candidate A = B0), so training starts from B0 and the S1 delta is exactly the
adapter's contribution; 6x lower training memory via row slicing
(`RN:109-110`); one loss and one readout for all three question types.

**We accept:** order sensitivity must be trained out (permutation
augmentation) and optionally averaged out at test time (P=2 or cyclic, each
permutation costs one extra branch per Choice question); a hard cap of 26
options and 10 score levels in v1 (single-token labels).

**We are now locked into:** training data shaped as per-question label
distributions, adapters that assume a frozen head, and teacher labels read
through the same label rows. Moving to a scalar head means a new training
recipe and 2 to 4 new runs.

**Revisit if:** Phase 0 M2 (Qwen3-1.7B-Base pilot) shows top-1 agreement
under reversal plus 3 permutations < 0.90 even with augmentation and P=2
pooling; then use a scalar head for Choice only, keeping this readout for
Score and Noul.

## Verification

Order suite in the eval harness (`REQ:234-237`): top-1 agreement and gold-rank
flip rate on Choice items with >= 3 options, per release. Failing if agreement
< 0.90 (MVP) or flip rate > 15% on the release model at its default P.
