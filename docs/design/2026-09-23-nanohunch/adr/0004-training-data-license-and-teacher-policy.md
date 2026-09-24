# ADR 0004: Released model and dataset use only redistributable data and open-weight teacher labels

**Status:** proposed (confirm the open-weight teacher's licence at selection)
**Date:** 2026-09-23
**Door:** one-way (expensive to reverse)
**Deciders:** @umang (author), sd-architect

## Context

The portfolio goal is a public model, code and eval (`RN:10-12`). Licences
measured today (`SM:197-213`): MMLU-Pro MIT; HotpotQA, BoolQ, SNLI
share-alike (CC-BY-SA); ANLI non-commercial (CC-BY-NC-4.0); pngwn
`typed-decisions-v2` card says CC-BY-SA-4.0 but the research notes say the
ticket component is CC-BY-NC-4.0 (`SM:45-49`), unresolved. Whether closed API
teachers' terms allow training and releasing a model on their outputs is
unverified (`REQ:456`, A8). A released dataset cannot be unpublished, and
relabelling after release costs budget.

## Decision

We will train the released model only on sources whose licences allow
redistribution of derivatives (MIT, Apache-2.0, CC-BY, CC-BY-SA), release the
dataset under CC-BY-SA-4.0 with per-row `source.license`, and produce bulk
training labels with an open-weight teacher under Apache-2.0 or MIT, read
through the nanohunch-fmt-v1 readout. Closed API teachers are used only for the
eval reference subset unless their terms are read and allow it. ANLI and
pngwn's data are eval-only.

## Alternatives considered

| Option | Why it lost |
|---|---|
| Cheap API teachers for bulk labels | Terms unverified (A8); if they forbid it, the dataset and model must be withdrawn or relabelled. Cheaper to start (no GPU teacher setup), so it stays allowed for private experiments |
| Include ANLI / pngwn tickets in training | NC licence (ANLI, measured) or unresolved licence (pngwn); training on pngwn would also contaminate the S5 out-of-distribution anchor (`REQ:248`) |
| Keep everything private | Removes the portfolio artefact (`REQ:424`) |

## Consequences

**We gain:** a model and dataset that can be published without a licence
question; soft labels from teacher logits in one forward (`RN:160-162`); every
row traceable to its licence.

**We accept:** a GPU line item for the open-weight teacher (2 GPU-hours or
less for 20k decisions, DERIVED in `REQ:216`, owned by `cost.md`); share-alike
on the released dataset; an open-weight teacher may be weaker than the closed
tier.

**We are now locked into:** the released dataset's licence and composition.
Changing teachers later means relabelling released rows.

**Revisit if:** a closed teacher's terms are read and explicitly permit
training and releasing a model on outputs, or pngwn's licence is confirmed
CC-BY-SA for the ticket component.

## Verification

A build step fails if any training row's `source.license` is not in the allow
list or any teacher label comes from a non-allowlisted teacher. Checked before
every training run and before publishing.
