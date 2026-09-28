"""Shared fixtures (plan Phase 2 step 1)."""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = ROOT / "runs" / "models" / "minicpm5-2b-base-raw"


@pytest.fixture(scope="session")
def tok():
    """The real MiniCPM5 tokenizer (transformers only, no weights, no GPU)."""
    if not MODEL.exists():
        pytest.skip(
            "run: uv run python -m bench.make_raw_model openbmb/MiniCPM5-2B-Base runs/models/minicpm5-2b-base-raw bos"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL))


def pytest_collection_modifyitems(config, items):
    if MODEL.exists():
        return
    skip = pytest.mark.skip(reason=f"{MODEL} missing")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
