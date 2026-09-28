# Small open base models for a System One scorer (as of 2026-09-23)

**Question:** best small open model to fine-tune into a typed-decision scorer on an M5 Mac with
under 200 USD. **Sources:** HF model cards and API (fetched 2026-09-23), pngwn controlled report,
on-machine MLX benchmarks with synthetic weights (`capacity.md`).

| Model | Params | Licence | Notes |
|---|---|---|---|
| Qwen3.5-4B(-Base) | 4B | Apache-2.0 | strongest knowledge (MMLU-Pro 79.1); hybrid Gated DeltaNet trains at 8 tok/s LoRA on the M5 (MEASURED, synthetic weights) |
| MiniCPM5-2B(-Base) | 2.52B | Apache-2.0 | plain Llama; 297 tok/s LoRA at 2k on the M5; W1 893 ms (MEASURED, synthetic) |
| Qwen3-4B-Base | 4.02B | Apache-2.0 | plain attention; 178 tok/s at 2k; challenger in the bake-off |
| Gemma 4 E2B/E4B, LFM2.5 | 2-8B | Apache / LFM | weaker in published comparisons |
| LFM2.5-Encoder-350M | 350M | LFM Open | best encoder at size; knowledge too thin for a main scorer |

**pngwn controlled report (2026-09-16):** readout head type changed state-derivable accuracy by
0.006; knowledge scaled with params (MMLU-Pro slice 0.625 at 4B vs 0.415 at 0.6B); letter
readout flipped 37.5% of rankings under reversed options; fine-tune truncated at 384 tokens lost
to the base above 2k tokens.

**Decision:** ADR-0001 (plain attention; MiniCPM5-2B MVP, Qwen3-4B if > 3 pts in the bake-off).
