"""bandad/asr-testset-kw-ja-v1の注釈を候補文字列に展開する。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from itertools import product
from math import prod

_TAG_START = re.compile(r"\(([A-Z]+)(?=\s|\)|$)")
_TAG = re.compile(r"\(([A-Z]+)(?:\s+([^()]*))?\)")
_OPTIONAL = {"D", "F", "VF", "VD"}
_VARIANT = {"V", "A", "VF", "VD"}
_IGNORED = {"L", "B", "C"}


class CandidateLimitError(ValueError):
    """厳密評価に必要な候補数が指定した上限を超えたことを示す。"""


def parse_reference(text: str) -> tuple[tuple[str, ...], ...]:
    """注釈を通常の文字列と候補の列に分解する。

    通常の括弧は本文として保持し、全角の注釈区切りにも対応する。
    不正なタグや、データセットの規約にない入れ子は例外にする。
    """
    if not isinstance(text, str):
        raise TypeError("reference must be a string")
    text = text.translate(str.maketrans({"（": "(", "）": ")", "｜": "|"}))
    chunks: list[tuple[str, ...]] = []
    cursor = 0
    for start in _TAG_START.finditer(text):
        if start.start() < cursor:
            continue
        match = _TAG.match(text, start.start())
        if match is None:
            raise ValueError(f"Malformed annotation at offset {start.start()}")
        tag, body = match.groups()
        if tag not in _OPTIONAL | _VARIANT | _IGNORED:
            raise ValueError(f"Unknown annotation tag: {tag}")
        chunks.append((text[cursor : start.start()],))
        if tag in _IGNORED:
            if body is not None:
                raise ValueError(f"{tag} must not have a text payload")
            choices = [""]
        else:
            if body is None:
                raise ValueError(f"{tag} requires a text payload")
            choices = body.split("|") if tag in _VARIANT else [body]
            choices = [choice if choice.strip() else "" for choice in choices]
            if not any(choices) or (tag != "A" and "" in choices):
                raise ValueError("Empty alternatives are only allowed alongside text in A")
            if tag not in _VARIANT and "|" in body:
                raise ValueError(f"{tag} does not accept alternatives")
            if tag in _OPTIONAL:
                choices.append("")
        chunks.append(tuple(dict.fromkeys(choices)))
        cursor = match.end()
    chunks.append((text[cursor:],))
    return tuple(chunks)


def reference_candidates(text: str, *, max_candidates: int | None = 100_000) -> Iterator[str]:
    """注釈の前後の本文を保持し、正解全文の候補を順に返す。

    max_candidatesを超える場合は近似せず例外にする。
    Noneを指定すると上限なしで探索する。候補全体はメモリに展開しない。
    """
    if max_candidates is not None and (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 1
    ):
        raise ValueError("max_candidates must be a positive integer or None")
    chunks = parse_reference(text)
    count = prod(map(len, chunks))
    if max_candidates is not None and count > max_candidates:
        raise CandidateLimitError(
            f"Reference has {count:,} candidates (limit {max_candidates:,}); "
            "evaluate shorter aligned segments or increase max_candidates"
        )
    for parts in product(*chunks):
        yield "".join(parts)
