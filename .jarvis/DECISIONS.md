# Decisions

Earlier decisions live in the design docs; this file indexes them and records new ones.

## 2026-09-24: execution mode is the jarvis contract
**Chose:** jarvis writes scaffolding, glue, every test and runs everything; the user types the
core from full-code hand-offs (at most ~40 lines each) that turn named tests green.
**Rejected:** chisel/pair (user types everything including tests and glue, hints only: about
160 h of hands-on time against ~57, with most of the difference being ceremony); agent writes
the core (defeats the learning goal).
**Because:** the time budget was the binding constraint in every review (`risks-premortem.md`
F1, stall at 60%). Removing ceremony keeps the learning (the core) and cuts the stall risk.
**Revisit if:** the user wants more struggle on a piece: "hints only" per step.

## 2026-09-24: Score label scheme is option (b) (ADR-0003 Amendment 1)
**Chose (user, 2026-09-24):** option (b): branch ends with the space id appended as an id, then
read bare-digit rows `0`..`9`.
**Rejected:** option (a) letters for Score (loses the base model's digit prior); the original
` 0`..` 9` (two ids each, P0-1 fail).
**Because:** measured 2026-09-24: ` 0`..` 9` are two ids in MiniCPM5 and Qwen3; bare digits are
single ids in both; `encode("Answer: 7") == encode("Answer: ") + encode("7")` in both, with
`"Answer: "` ending in the space id (242 MiniCPM5, 220 Qwen3). So (b) is exactly the natural
tokenization.
**Revisit if:** a candidate tokenizer breaks the equality, or the Phase 3 bake-off shows Score
calibration far worse than Choice.

## Index of earlier decisions (design docs)
- ADR-0001 + amendment: plain-attention base; MiniCPM5-2B-Base MVP, Qwen3-4B-Base if > 3 pts.
- ADR-0002: restricted label-logit readout on the frozen LM head.
- ADR-0003: prompt format `nanohunch-fmt-v1`, split tokenization, frozen after Phase 2.
- ADR-0004: releasable teachers only (DeepSeek V4.1 Flash, Qwen3.6-35B-A3B); NC data eval-only.
- `risks.md` R21-R23: claim is "minimal", 1,000-line core budget, external eval sets eval-only.
