# ADR 0004: Released model and dataset use only redistributable data and open-weight teacher labels

**Status:** accepted as amended (Amendment 2 supersedes Amendment 1's teacher list)
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

## Amendment 1 (2026-09-24, aligns this ADR with `risks.md` R4, R16 and R22)

- Bulk labels come from two teachers via OpenRouter logprobs, with no GPU line:
  **DeepSeek V4.1 Flash** (API; ToS 4.2(3) permits training on and
  distributing outputs if published outputs are marked AI-generated; confirm
  under Q6 before bulk labelling, else remove it from `RELEASABLE_TEACHERS`)
  and **Qwen3.6-35B-A3B** (Apache-2.0 open weights, served by an OpenRouter
  host). This is the "terms are read and allow it" case of the Decision above.
- OpenAI, Anthropic and Jev outputs never touch training, selection,
  calibration or published labels.
- Eval-only, never trained on: ANLI and all pngwn artefacts (NC treatment),
  SemIf authored144 and JevBench public items (R22, excluded by
  `configs/eval_only_hashes.txt`).

## Amendment 2 (2026-09-25, Phase 0 gate; supersedes Amendment 1's teacher list)

The author has no OpenRouter account, so the API teachers are dropped. Bulk labels and the
synthetic-state generator run **open-weight models locally in MLX**, which is this ADR's original
decision:

- **Teachers:** `lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit` (Apache-2.0) and
  `unsloth/Qwen3.6-35B-A3B-UD-MLX-3bit` (Apache-2.0). Two model families, which reduces
  correlated teacher error (R11). `RELEASABLE_TEACHERS = {"gemma-4-26b-a4b", "qwen3.6-35b-a3b"}`.
- **Readout:** full-vocabulary softmax at the answer position, summed over single-id spellings of
  each label, then renormalized. This is exact, with no top-20 truncation. Measured on 50 items:
  candidate mass >= 0.997 for both (`reports/phase0.md` P0-7').
- **Closed APIs** (Anthropic, OpenAI, Gemini, Jev): eval-only reference rows at most. Checked
  2026-09-24: the Gemini API terms forbid using the service to develop competing models and forbid
  extracting or replicating its models, which is not an explicit permission to distill.
- **Q6** (DeepSeek terms) is moot. No API spend is planned for labels.
- **Consequence for the plan:** `label.py` becomes a local MLX teacher runner with the same cache
  contract (one JSONL line per call, allowlisted fields, run date recorded) instead of HTTP glue.
  Each teacher labels a dataset in one pass: same-day reruns are bit-identical, but a 0.025 drift
  against a 12-hour-old run was observed with an identical prompt (cause unknown).
