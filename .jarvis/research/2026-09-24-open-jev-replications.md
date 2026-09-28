# Open Jev replications (as of 2026-09-24)

**Question:** which open System One / Jev replications exist, and what should a minimal one
borrow or compare against? **Method:** GitHub search, then the source of six repos read by two
research agents; JevBench README; SemIf `docs/RESULTS.md`. Nothing was run by us.

| Repo (pinned sha) | Approach | Core size | Trained | Numbers they publish |
|---|---|---|---|---|
| TheoLeeCJ/SemIf-OpenJev @23cf1f3, MIT | frozen Qwen3.5-4B, letter logits A-P, shared-state batching, MLX backend | ~640 lines minimal path | no; T per workload | authored144 family bal. acc: Qwen3.5-4B 0.813, MiniCPM5-2B 0.686, Qwen3-0.6B 0.440; TypeSafe subset 0.845 vs Jev 0.883; JevBench v1.3 73.1 (Jev 74.4) |
| Mapika/decider @b44b4c9, Apache-2.0 | full fine-tune Qwen3.5 0.8B-35B, restricted letter-slot logits, many questions per pass | 759 core / 11.9k total | yes: 1.47M examples, Qwen3.5-27B teacher, 5.3 h GH200 | held-out acc 0.755 (2B), 0.788 (4B); ECE 0.037 in-task vs 0.121 zero-shot |
| kshetrajna12/reflex @231f896, MIT | frozen Qwen3.5-4B, letter logits, 2 orders in one pass | ~1,750 core | LoRA tried, every adapter rejected | JevBench v1.4 rank 5 (53.99) |
| wfzyx/von @1d86116, Apache-2.0 | ModernBERT-large, scalar per [MASK] option marker, order-invariant masks | ~780 core | yes, partly Jev-distilled | self-contradictory claims |
| Colvin0315/MiniSystemOne @016c6dd, Apache-2.0 | 27M encoder from scratch | ~3.5k | synthetic only | historical numbers, no weights |
| rupeshpoojary9/poorjev, MIT | DeBERTa NLI wrapper | ~785 | no | Banking77 0.656 vs Jev 0.812 |
| fstandhartinger/jevbench @2fa63fa, MIT | benchmark: 534 decisions, public half + sealed | n/a | n/a | public items usable for self-run evals; public half can be trained on, so eval-only |

**Takeaways for nanohunch**
- Minimal is unclaimed: trained repos are large, small repos are untrained.
- MiniCPM5-2B frozen is 12.7 pts below frozen Qwen3.5-4B on authored144: expect the Phase 3
  base rule to fire.
- LoRA can hurt general judgement (reflex); large-scale full fine-tuning helped (decider).
  Keep the Phase 5 kill rule.
- Borrow: SemIf `mlx_backend.py` (cache copy per question), decider `slot_logits`, reflex
  `train/calibrate.py` (proper-score loss), MiniSystemOne `calibrate_temperature.py`
  (T per option count), von masks (later list).

**Staleness:** fast-moving field (new repos daily in the week after launch). Re-run the GitHub
search with a 30-day window before Phase 6's comparison table.
