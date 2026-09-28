"""Model-shape tests keep the scorer's hidden-state/head wiring explicit."""

import sys
from types import ModuleType, SimpleNamespace


def _scorer(monkeypatch, model):
    """Import the engine behind fake MLX seams so path tests also run headlessly."""
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    core = ModuleType("mlx.core")
    core.array = type("Array", (), {})
    core.uint32 = object()
    core.float32 = object()
    mlx.core = core
    utils = ModuleType("mlx.utils")
    utils.tree_map = lambda fn, values: values
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.__path__ = []
    mlx_lm.load = lambda *args, **kwargs: (model, object())
    models = ModuleType("mlx_lm.models")
    models.__path__ = []
    cache = ModuleType("mlx_lm.models.cache")
    cache.can_trim_prompt_cache = lambda value: True
    cache.make_prompt_cache = lambda value: []
    cache.trim_prompt_cache = lambda *args: None
    for name, module in {
        "mlx": mlx,
        "mlx.core": core,
        "mlx.utils": utils,
        "mlx_lm": mlx_lm,
        "mlx_lm.models": models,
        "mlx_lm.models.cache": cache,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    # raising=False is a no-op when "engine" was never imported yet, so monkeypatch never
    # tracks it for cleanup and the fake-mlx-bound module leaks into later tests. Force
    # monkeypatch to record a restore-to-absent action regardless of prior state.
    if "engine" in sys.modules:
        monkeypatch.delitem(sys.modules, "engine")
    else:
        monkeypatch.setitem(sys.modules, "engine", None)
        del sys.modules["engine"]

    import engine

    return engine.MLXBranchScorer("fixture")


def test_scorer_resolves_qwen35_nested_text_model_and_tied_head(monkeypatch):
    """Qwen3.5 wraps the text backbone below language_model and ties its token head."""
    head_weight = object()
    backbone = SimpleNamespace(embed_tokens=SimpleNamespace(weight=head_weight))
    text_model = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=True),
        model=backbone,
    )
    model = SimpleNamespace(language_model=text_model)
    scorer = _scorer(monkeypatch, model)

    assert scorer._backbone is backbone
    assert scorer._head_weight is head_weight


def test_scorer_keeps_flat_text_model_path(monkeypatch):
    """Existing Llama-shaped models still resolve through their flat wrapper."""
    head_weight = object()
    backbone = SimpleNamespace(embed_tokens=SimpleNamespace(weight=head_weight))
    model = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=True),
        model=backbone,
    )
    scorer = _scorer(monkeypatch, model)

    assert scorer._backbone is backbone
    assert scorer._head_weight is head_weight
