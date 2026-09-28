"""日本語・英数字の境界にある空白を正規化する。"""

from __future__ import annotations

import re
import unicodedata

JP_CHARS = r"一-龥々〆ヵヶぁ-ゖゝゞァ-ヺヽヾー"
ASCII_ALNUM = r"A-Za-z0-9"

JP_PUNCTS = "、。，．・！？：；…‥"
ASCII_PUNCTS = ",.!?:;"
OPEN_BRACKETS = "（［｛〈《「『【〔("
CLOSE_BRACKETS = "）］｝〉》」』】〕)"
CURRENCY = "¥￥$"
NUM_UNITS = "%％°℃"


def _escape(chars: str) -> str:
    return re.escape(chars)


RE_WS = re.compile(r"\s+")
RE_JP_JP_SPACE = re.compile(rf"(?<=[{JP_CHARS}])\s+(?=[{JP_CHARS}])")
RE_JP_ASCII_SPACE = re.compile(rf"(?<=[{JP_CHARS}])\s+(?=[{ASCII_ALNUM}])")
RE_ASCII_JP_SPACE = re.compile(rf"(?<=[{ASCII_ALNUM}])\s+(?=[{JP_CHARS}])")
RE_SPACE_BEFORE_PUNCT = re.compile(rf"\s+([{_escape(JP_PUNCTS + ASCII_PUNCTS)}])")
RE_JA_OPEN_BRACKET_SPACE = re.compile(rf"(?<=[{JP_CHARS}])\s+(?=[{_escape(OPEN_BRACKETS)}])")
RE_JA_CLOSE_BRACKET_SPACE = re.compile(rf"(?<=[{_escape(CLOSE_BRACKETS)}])\s+(?=[{JP_CHARS}])")
RE_SPACE_AFTER_OPEN_BRACKET = re.compile(rf"([{_escape(OPEN_BRACKETS)}])\s+")
RE_SPACE_BEFORE_CLOSE_BRACKET = re.compile(rf"\s+([{_escape(CLOSE_BRACKETS)}])")
RE_SPACE_AFTER_JP_PUNCT = re.compile(
    rf"([{_escape(JP_PUNCTS)}])\s+(?=[{JP_CHARS}{_escape(OPEN_BRACKETS)}])",
)
RE_ALNUM_CONNECTOR = re.compile(
    rf"(?<=[{ASCII_ALNUM}])\s*([\-_/.:])\s*(?=[{ASCII_ALNUM}])",
)
RE_SPACE_BEFORE_SIGNED_NUM = re.compile(
    rf"(?<=[{JP_CHARS}{ASCII_ALNUM}{_escape(CLOSE_BRACKETS)}])\s+(?=[+\-−]\d)",
)
RE_SIGN_SPACE_NUM = re.compile(r"([+\-−])\s+(?=\d)")
RE_NUM_UNIT = re.compile(rf"(?<=\d)\s+([{_escape(NUM_UNITS)}])")
RE_CURRENCY_NUM = re.compile(rf"([{_escape(CURRENCY)}])\s+(?=\d)")
RE_JP_CURRENCY = re.compile(rf"(?<=[{JP_CHARS}])\s+(?=[{_escape(CURRENCY)}])")
RE_CURRENCY_JP = re.compile(rf"(?<=[{_escape(CURRENCY)}\d])\s+(?=[{JP_CHARS}])")


def normalize_ja_simple(text: str) -> str:
    """日本語・句読点・単位の前後の空白と文字幅を正規化する。"""
    normalized = RE_WS.sub(" ", text).strip()
    if not normalized:
        return normalized

    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.replace("\u3000", " ")
    normalized = RE_WS.sub(" ", normalized).strip()
    if not normalized:
        return normalized

    normalized = RE_SPACE_AFTER_OPEN_BRACKET.sub(r"\1", normalized)
    normalized = RE_SPACE_BEFORE_CLOSE_BRACKET.sub(r"\1", normalized)
    normalized = RE_SPACE_BEFORE_PUNCT.sub(r"\1", normalized)
    normalized = RE_SPACE_AFTER_JP_PUNCT.sub(r"\1", normalized)
    normalized = RE_JA_OPEN_BRACKET_SPACE.sub(r"", normalized)
    normalized = RE_JA_CLOSE_BRACKET_SPACE.sub(r"", normalized)

    normalized = RE_JP_JP_SPACE.sub("", normalized)
    normalized = RE_JP_ASCII_SPACE.sub("", normalized)
    normalized = RE_ASCII_JP_SPACE.sub("", normalized)

    normalized = RE_ALNUM_CONNECTOR.sub(r"\1", normalized)
    normalized = RE_SPACE_BEFORE_SIGNED_NUM.sub("", normalized)
    normalized = RE_SIGN_SPACE_NUM.sub(r"\1", normalized)
    normalized = RE_NUM_UNIT.sub(r"\1", normalized)
    normalized = RE_CURRENCY_NUM.sub(r"\1", normalized)
    normalized = RE_JP_CURRENCY.sub("", normalized)
    normalized = RE_CURRENCY_JP.sub("", normalized)

    return normalized
