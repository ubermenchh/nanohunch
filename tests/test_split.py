"""dataset.assign_split: hashed, group-level, deterministic splits (plan Phase 3 steps 2-3; R12)."""

import pytest

FRAC = {"train": 0.8, "cal": 0.08, "test": 0.12}


def test_assign_split_deterministic():
    from dataset import assign_split

    keys = [f"g{i}" for i in range(1000)]
    first = [assign_split(k, "boolq", FRAC, "test-salt") for k in keys]
    assert first == [assign_split(k, "boolq", FRAC, "test-salt") for k in keys]
    # pinned values: a change to the hash recipe would silently reshuffle every split
    assert assign_split("g0", "boolq", FRAC, "test-salt") == "train"
    assert assign_split("g28", "boolq", FRAC, "test-salt") == "test"
    assert assign_split("g35", "boolq", FRAC, "test-salt") == "cal"
    assert assign_split("g35", "arc", FRAC, "test-salt") == "test"  # the source is part of the hash


def test_no_group_in_two_splits():
    from dataset import assign_split

    seen: dict[str, set[str]] = {}
    for i in range(10_000):
        g = f"grp{i % 2000}"  # 5 items per group; the item id plays no part
        seen.setdefault(g, set()).add(assign_split(g, "hotpot", FRAC, "test-salt"))
    assert all(len(s) == 1 for s in seen.values())


def test_fractions_within_1pct_at_10k():
    from dataset import assign_split

    got = [assign_split(f"g{i}", "boolq", FRAC, "test-salt") for i in range(10_000)]
    for name, frac in FRAC.items():
        assert abs(got.count(name) / len(got) - frac) < 0.01, name


def test_bad_fractions_raise():
    from dataset import assign_split

    with pytest.raises(ValueError, match="sum to"):
        assign_split("g0", "boolq", {"train": 0.8, "test": 0.1}, "test-salt")
