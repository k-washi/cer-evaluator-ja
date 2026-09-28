"""読みへの変換を使わず、CER向けに数値周辺の表記を統一する。

speech-ja-text-frontendのmatcher/text_cer.pyを参考にした追加規則。
桁区切りは既存の数値正規化に任せ、カンマの有無で結果が変わらないようにする。
"""

from __future__ import annotations

import re

from .common import Range
from .digit import DEFAULT_COUNTER_SUFFIXES, NUM_OR_DECIMAL_TOKEN_RE

# 日本語に隣接した番号にも対応し、英数字の型番・小数・日付の一部は拾わない。
_LEFT = r"(?<![0-9A-Za-z_+./-])"
_RIGHT = r"(?![0-9A-Za-z_./-])"
_PHONE = re.compile(rf"{_LEFT}(?:\+[0-9]{{1,3}}-|0)[0-9]*(?:-[0-9]+){{1,3}}{_RIGHT}")
_POSTAL = re.compile(rf"{_LEFT}([0-9]{{3}})-([0-9]{{4}}){_RIGHT}")
# 電話番号・先頭0の郵便番号は数量ではない。区切りのない表記も同様に保護する。
_IDENTIFIER = re.compile(rf"{_LEFT}(?:0(?:[0-9]{{6}}|[0-9]{{9,10}})|\+[0-9]{{8,15}}){_RIGHT}")
_LENGTH_UNITS = (
    (
        re.compile(rf"({NUM_OR_DECIMAL_TOKEN_RE})(?:センチメートル|センチ|シーエム|cm)", re.I),
        r"\1cm",
    ),
    (re.compile(rf"({NUM_OR_DECIMAL_TOKEN_RE})(?:ミリメートル|ミリ|エムエム|mm)", re.I), r"\1mm"),
)
_SUFFIXES = "|".join(
    re.escape(suffix)
    for suffix in sorted(
        {
            *DEFAULT_COUNTER_SUFFIXES,
            "%",
            "円",
            "ドル",
            "ユーロ",
            "ポンド",
            "cm",
            "mm",
            "km",
            "kg",
            "m",
            "g",
        },
        key=lambda suffix: (-len(suffix), suffix),
    )
)
_RANGE = re.compile(
    rf"(?<![0-9a-z.+-])"
    rf"({NUM_OR_DECIMAL_TOKEN_RE})(?:\s*({_SUFFIXES}))?"
    rf"\s*(?:から|〜|~)\s*"
    rf"({NUM_OR_DECIMAL_TOKEN_RE})(?:\s*({_SUFFIXES}))?"
    rf"(?:\s*まで)?(?![0-9a-z.])"
)


def prepare_numeric_surfaces(text: str) -> str:
    """ゼロ・長さの単位・番号のハイフンを数値解析の前に統一する。

    ゼロの置換は参照実装と同じ文字列置換。イチなどの読みは変換しない。
    国内電話は10/11桁、国際電話は+に続く8〜15桁に限定して区切りを除く。
    郵便番号は3桁-4桁。郵便番号も区切りのない表記と同じ数値規則に従う。
    """
    text = text.replace("ゼロ", "0")
    for pattern, replacement in _LENGTH_UNITS:
        text = pattern.sub(replacement, text)

    def phone(match: re.Match[str]) -> str:
        compact = match.group().replace("-", "")
        valid = 8 <= len(compact) - 1 <= 15 if compact.startswith("+") else len(compact) in (10, 11)
        return compact if valid else match.group()

    return _POSTAL.sub(r"\1\2", _PHONE.sub(phone, text))


def identifier_ranges(text: str) -> list[Range]:
    """電話番号などの先頭0と数字列を数量への変換から保護する。"""
    return [match.span() for match in _IDENTIFIER.finditer(text)]


def normalize_numeric_ranges(text: str) -> str:
    """数値範囲の「から」「〜」「~」を「〜」に揃え、直後の「まで」を除く。

    左右の数値・単位は保持する。減算やスコアに使うハイフンは範囲にしない。
    負数の一部分だけを誤変換しないため、符号付き範囲は対象外とする。
    """

    def replace(match: re.Match[str]) -> str:
        left, left_unit, right, right_unit = match.groups()
        return f"{left}{left_unit or ''}〜{right}{right_unit or ''}"

    return _RANGE.sub(replace, text)
