"""正規化処理で共通に使用する補助関数。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Range: TypeAlias = tuple[int, int]

REPEATED_LONG_VOWEL_RE = re.compile(r"ー{2,}")


def normalize_long_vowel_repeats_chunk(text: str) -> str:
    """連続する長音記号を1文字にまとめる。"""
    if not text:
        return text

    normalized = text.replace("ｰ", "ー")
    return REPEATED_LONG_VOWEL_RE.sub("ー", normalized)


def delete_kutouten(text: str) -> str:
    """句読点「、。!?」を削除する。"""
    return text.replace("、", "").replace("。", "").replace("!", "").replace("?", "")


def apply_to_unprotected_ranges(
    text: str,
    protected_ranges: Sequence[Range],
    transform: Callable[[str], str],
) -> str:
    """保護範囲の外側だけに変換関数を適用する。"""
    if not protected_ranges:
        return transform(text)

    parts: list[str] = []
    cursor = 0

    for start, end in protected_ranges:
        if cursor < start:
            parts.append(transform(text[cursor:start]))

        parts.append(text[start:end])
        cursor = end

    if cursor < len(text):
        parts.append(transform(text[cursor:]))

    return "".join(parts)
