# Convert an adapter saved by run_decider.py (MLX .npz) to the PEFT format, so transformers + peft can load it.
#   uv run python to_peft.py adapter_X.npz out_dir
# Our LoRALinear computes base(x) + (alpha / rank) * (x @ lora_a) @ lora_b, which is PEFT's LoRA with lora_A = lora_a.T, lora_B = lora_b.T.
import json
import sys
from pathlib import Path

import mlx.core as mx

adapter, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.load(open(adapter.with_suffix(".json")))
out.mkdir(parents=True, exist_ok=True)
weights = {}
for key, w in mx.load(str(adapter)).items():  # model.layers.0.self_attn.q_proj.lora_a, shape (in, rank)
    name, part = key.rsplit(".", 1)
    weights[f"base_model.model.{name}.lora_{part[-1].upper()}.weight"] = mx.contiguous(w.T)  # PEFT: (rank, in) and (out, rank)
mx.save_safetensors(str(out / "adapter_model.safetensors"), weights, metadata={"format": "pt"})
json.dump(
    dict(
        peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=cfg["model"], r=cfg["rank"], lora_alpha=cfg["alpha"],
        lora_dropout=0.0, bias="none", target_modules=[t.split(".")[-1] for t in cfg["targets"]], inference_mode=True,
    ),
    open(out / "adapter_config.json", "w"), indent=2,
)
print(f"wrote {len(weights)} tensors to {out}")
