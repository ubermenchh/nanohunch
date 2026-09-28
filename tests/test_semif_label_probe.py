"""The parity probe must test the reference bare label tokens, not invent mappings."""

import pytest

from bench.semif_label_probe import bare_choice_ids


def test_bare_choice_ids_use_exact_uppercase_single_tokens():
    class Tok:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [ord(text)]

    assert bare_choice_ids(Tok(), 3) == (ord("A"), ord("B"), ord("C"))


def test_bare_choice_ids_reject_multitoken_label():
    class Tok:
        def encode(self, text, *, add_special_tokens):
            return [1, 2] if text == "B" else [ord(text)]

    with pytest.raises(ValueError, match="not one token"):
        bare_choice_ids(Tok(), 3)
