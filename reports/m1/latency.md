## M1 latency (bench/latency_p2.py, W0/W1/W2, plan Phase 2 step 11)

## Latency: openbmb/MiniCPM5-2B-Base (engine (branching), batch 1, bf16)

| workload | state tokens | questions | median ms | synthetic ms | ratio |
|---|---|---|---|---|---|
| W0 | 256 | 4 | 297 | 318 | 0.93x |
| W1 | 1024 | 16 | 1261 | 893 | 1.41x |
| W2 | 8192 | 16 | 4386 | 6600 | 0.66x |

## Latency: Qwen/Qwen3-4B-Base (engine (branching), batch 1, bf16)

| workload | state tokens | questions | median ms | synthetic ms | ratio |
|---|---|---|---|---|---|
| W0 | 256 | 4 | 596 | 318 | 1.88x |
| W1 | 1024 | 16 | 2060 | 893 | 2.31x |
| W2 | 8192 | 16 | 7654 | 6600 | 1.16x |

## Latency: Qwen/Qwen3.5-4B-Base (re-encode, batch 1, bf16)

| workload | state tokens | questions | median ms | synthetic ms | ratio |
|---|---|---|---|---|---|
| W0 | 256 | 4 | 972 | 318 | 3.06x |
| W1 | 1024 | 16 | 12923 | 893 | 14.47x |
| W2 | 8192 | 16 | 126528 | 6600 | 19.17x |
