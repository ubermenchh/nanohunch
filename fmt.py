from dataclasses import dataclass
from typing import Literal

QType = Literal["choice", "score", "noul"]
Perm = tuple[int, ...]  # perm[display_pos] = canonical option index
FORMAT_VERSION = "nanohunch-fmt-v1"

PREAMBLE = "You will answer questions about the STATE. Answer with the label of exactly one option.\n\n"


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    type: QType
    text: str
    options: tuple[str, ...]
    values: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Branch:
    question_id: str
    qtype: QType
    perm: Perm
    token_ids: tuple[int, ...]
    label_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Rendered:
    format_version: str
    prefix_ids: tuple[int, ...]
    branches: tuple[Branch, ...]
    truncated: bool


class TooManyOptions(ValueError): ...


class StateTooLong(ValueError): ...


class LabelNotSingleToken(ValueError): ...


def label_strings(qtype: QType, n: int) -> list[str]:
    if qtype == "choice":
        return [f" {chr(65 + i)}" for i in range(n)]
    if qtype == "score":
        return [str(v) for v in range(n)]  # Amendment 1 (b): bare digits, read after an appended space id
    return [" yes", " no"]


def label_vocab(tokenizer, qtype: QType, n: int) -> tuple[int, ...]:
    ids = []
    for s in label_strings(qtype, n):
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) != 1:
            raise LabelNotSingleToken(f"{s!r} -> {enc}")
        ids.append(enc[0])
    return tuple(ids)


def permutations_for(n_options: int, n_perms: int) -> list[Perm]:
    ident = tuple(range(n_options))
    cands = [ident, ident[::-1]] + [tuple((d + s) % n_options for d in ident) for s in range(1, n_options)]
    perms = list(dict.fromkeys(cands))
    if n_perms > len(perms):
        raise ValueError(f"{n_perms} perms requested, only {len(perms)} distinct for {n_options} options")
    return perms[:n_perms]


def _space_id(tokenizer) -> int:
    ctx = tokenizer.encode("Answer:", add_special_tokens=False)
    with_space = tokenizer.encode("Answer: ", add_special_tokens=False)
    if len(with_space) != len(ctx) + 1 or with_space[: len(ctx)] != ctx:
        raise LabelNotSingleToken(f"'Answer: ' -> {with_space}: no single space id")
    return with_space[-1]


def _validate(q: Question) -> None:
    n = len(q.options)
    if q.type == "choice" and n > 26:
        raise TooManyOptions(f"{q.id}: {n} options > 26")
    if q.type == "score":
        if n > 10:
            raise TooManyOptions(f"{q.id}: {n} levels > 10")
        if len(q.values) != n or list(q.values) != sorted(set(q.values)) or not all(0 <= v <= 9 for v in q.values):
            raise ValueError(f"{q.id}: score values must be ascending, distinct, 0..9, one per option")
    if q.type == "noul" and q.options != ("yes", "no"):
        raise ValueError(f"{q.id}: noul options must be ('yes', 'no')")


def _option_lines(q: Question, shown: list[str]) -> str:
    if q.type == "choice":
        return "\n".join(f"{chr(65 + d)}. {o}" for d, o in enumerate(shown))
    if q.type == "score":
        return "\n".join(f"{v}: {o}" for v, o in zip(q.values, shown, strict=True))
    return "(yes/no)"


def render(tokenizer, state, questions, *, n_perms: int, max_context: int) -> Rendered:
    prefix_ids = tuple(tokenizer.encode(f"{PREAMBLE}### STATE\n{state}\n### END STATE\n\n"))
    branches = []
    for q in questions:
        _validate(q)
        n = len(q.options)
        vocab = label_vocab(tokenizer, q.type, 10 if q.type == "score" else n)
        labels = tuple(vocab[v] for v in q.values) if q.type == "score" else vocab
        for perm in permutations_for(n, n_perms) if q.type == "choice" else [tuple(range(n))]:
            shown = [q.options[perm[d]] for d in range(n)]
            ids = tokenizer.encode(
                f"### QUESTION\n{q.text}\n{_option_lines(q, shown)}\nAnswer:", add_special_tokens=False
            )
            if q.type == "score":
                ids = ids + [_space_id(tokenizer)]
            branches.append(Branch(q.id, q.type, perm, tuple(ids), labels))
    longest = max(len(b.token_ids) for b in branches)
    if len(prefix_ids) + longest > max_context:
        raise StateTooLong(f"{len(prefix_ids)} + {longest} > {max_context}")
    return Rendered(FORMAT_VERSION, prefix_ids, tuple(branches), False)
