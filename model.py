import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

def label_logits(hidden_last: mx.array, head_weight: mx.array, label_ids: mx.array) -> mx.array:
    return hidden_last.astype(mx.float32) @ head_weight[label_ids].astype(mx.float32).T

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        out_dims, in_dims = linear.weight.shape
        bound = 1 / in_dims**0.5
        self.lora_a = mx.random.uniform(-bound, bound, (in_dims, rank))
        self.lora_b = mx.zeros((rank, out_dims))
        self.scale = alpha / rank

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x) + (self.scale * (x @ self.lora_a) @ self.lora_b).astype(x.dtype)

def load_model(path: str):
    return load(path) # (model, tokenizer)

def add_lora(model, rank: int, alpha: float, num_layers: int, targets=("self_attn.q_proj", "self_attn.v_proj")):
    model.freeze()
    for layer in model.layers[-num_layers:]:
        for path in targets:
            block, name = path.split(".") # mlp.gate_proj -> layer.mlp.gate_proj
            parent = getattr(layer, block)
            setattr(parent, name, LoRALinear(getattr(parent, name), rank, alpha))

def answer_logits(model, ids: list[int], label_ids: list[int]) -> mx.array:
    h = model.model(mx.array(ids)[None])[0, -1]
    return label_logits(h[None], model.model.embed_tokens.weight, mx.array(label_ids))[0]