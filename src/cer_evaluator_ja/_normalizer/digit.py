"""日本語の数値表記を正規化する。

漢数字・算用数字を統一し、固有名詞と保護語句は変換しない。
正規表現をキャッシュし、繰り返しの正規化で再利用する。
元実装: data/speech-ja-text-frontend/ja_text_frontend/normalizer/digit.py。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from cer_evaluator_ja._normalizer.common import Range, apply_to_unprotected_ranges

try:
    from sudachipy import tokenizer as sudachi_tokenizer
except Exception:
    sudachi_tokenizer = None

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sudachipy.sudachipy import Morpheme

MODULE_DIR = Path(__file__).parent
DICT_DIR = MODULE_DIR / "dict"

with (DICT_DIR / "jyosuushi.json").open(encoding="utf-8") as file:
    DEFAULT_COUNTER_SUFFIXES: list[str] = json.load(file)

assert DEFAULT_COUNTER_SUFFIXES, "Counter suffixes list is empty. Check jyosuushi.json."

with (DICT_DIR / "protect_phrases_kansuuji.json").open(encoding="utf-8") as file:
    DEFAULT_PROTECTED_PHRASES: list[str] = json.load(file)

DEFAULT_PROTECT_POS_KEYWORDS: tuple[str, ...] = ("固有名詞",)
BIG_UNIT_THRESHOLD = 10_000
MAX_LEXICAL_AGE = 150
LEXICAL_NUMERIC_COUNTER_SUFFIXES = frozenset(
    {
        "人",
        "人組",
        "人組み",
        "人ぐみ",
        "歳",
        "歳児",
        "才",
        "日",
        "番",
    },
)


@dataclass(slots=True)
class TokenSpan:
    """形態素の文字位置と品詞情報を保持する。"""

    surface: str
    start: int
    end: int
    pos: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NumericPatterns:
    """数値正規化で共有するコンパイル済み正規表現を保持する。"""

    date_ymd: re.Pattern[str]
    date_ym: re.Pattern[str]
    date_md: re.Pattern[str]
    quarter: re.Pattern[str]
    score: re.Pattern[str]
    range_expr: re.Pattern[str]
    decimal_counter: re.Pattern[str]
    decimal_plain: re.Pattern[str]
    counter: re.Pattern[str]
    plain: re.Pattern[str]


KANJI_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "壱": 1,
    "壹": 1,
    "二": 2,
    "弐": 2,
    "弍": 2,
    "貳": 2,
    "三": 3,
    "参": 3,
    "參": 3,
    "四": 4,
    "五": 5,
    "伍": 5,
    "六": 6,
    "陸": 6,
    "陆": 6,
    "七": 7,
    "漆": 7,
    "柒": 7,
    "八": 8,
    "捌": 8,
    "九": 9,
    "玖": 9,
}

SMALL_UNITS = {
    "十": 10,
    "拾": 10,
    "什": 10,
    "百": 100,
    "陌": 100,
    "佰": 100,
    "千": 1000,
    "阡": 1000,
    "仟": 1000,
}

BIG_UNITS = {
    "万": 10**4,
    "億": 10**8,
    "兆": 10**12,
    "京": 10**16,
    "垓": 10**20,
    "𥝱": 10**24,
}

BIG_UNIT_ORDER = ["𥝱", "垓", "京", "兆", "億", "万"]

NUM_CHARS = "0-9〇零一二三四五六七八九十百千万億兆京垓𥝱壱壹弐弍貳参參肆伍陸陆漆柒捌玖拾什陌佰阡仟"

GROUPED_ARABIC_INT_RE = r"\d{1,3}(?:,\d{3})+"
UNGROUPED_ARABIC_INT_RE = r"\d+"
DECIMAL_TOKEN_RE = rf"(?:{GROUPED_ARABIC_INT_RE}|{UNGROUPED_ARABIC_INT_RE})\.\d+"
NUM_TOKEN_RE = rf"[{NUM_CHARS},]+"
NUM_OR_DECIMAL_TOKEN_RE = rf"(?:{DECIMAL_TOKEN_RE}|{NUM_TOKEN_RE})"
STRICT_THOUSANDS_COMMA_RE = re.compile(r"^\d{1,3}(,\d{3})+$")
ARABIC_COMMA_CHUNK_RE = re.compile(r"\d[\d,]*")
ASCII_INTEGER_RE = re.compile(r"\d+")
DECIMAL_SURFACE_RE = re.compile(DECIMAL_TOKEN_RE)
NUMERIC_SURFACE_RE = re.compile(rf"[{NUM_CHARS}]+")
NUMERIC_CANDIDATE_RE = re.compile(rf"[{NUM_CHARS}]")
NORMALIZED_QUARTER_RE = re.compile(r"第\d+四半期")


def strip_valid_thousands_commas(text: str) -> str:
    """正しい3桁区切りのカンマだけを削除する。"""

    def repl(match: re.Match[str]) -> str:
        chunk = match.group(0)
        if "," in chunk and STRICT_THOUSANDS_COMMA_RE.fullmatch(chunk):
            return chunk.replace(",", "")
        return chunk

    return ARABIC_COMMA_CHUNK_RE.sub(repl, text)


def build_token_spans(text: str, morphemes: Sequence[Morpheme]) -> list[TokenSpan]:
    """各形態素に対応する原文中の文字範囲を求める。"""
    spans: list[TokenSpan] = []
    cursor = 0

    for morpheme in morphemes:
        surface = morpheme.surface()
        if not surface:
            continue

        start = text.find(surface, cursor)
        if start < 0:
            start = cursor
        end = start + len(surface)
        cursor = end

        try:
            pos = tuple(morpheme.part_of_speech())
        except Exception:
            pos = ()

        spans.append(TokenSpan(surface=surface, start=start, end=end, pos=pos))

    return spans


def is_protected_token(
    token: TokenSpan,
    protect_pos_keywords: Sequence[str] = DEFAULT_PROTECT_POS_KEYWORDS,
) -> bool:
    """形態素を正規化の対象外にすべきか判定する。"""
    if not token.pos:
        return False
    pos_text = " ".join(token.pos)
    return any(keyword in pos_text for keyword in protect_pos_keywords)


@cache
def _prepare_unique_longest_first(values: tuple[str, ...]) -> tuple[str, ...]:
    """空文字と重複を除き、文字数の降順で返す。"""
    return tuple(sorted({value for value in values if value}, key=len, reverse=True))


def find_phrase_ranges(text: str, phrases: Sequence[str]) -> list[Range]:
    """指定語句が現れるすべての文字範囲を求める。"""
    ranges: list[Range] = []
    prepared_phrases = _prepare_unique_longest_first(tuple(phrases))

    for phrase in prepared_phrases:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            ranges.append((index, index + len(phrase)))
            start = index + 1

    return ranges


def merge_ranges(ranges: Sequence[Range]) -> list[Range]:
    """重複または隣接する文字範囲を統合する。"""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges)
    merged: list[Range] = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def collect_protected_ranges(
    text: str,
    token_spans: Sequence[TokenSpan],
    protected_phrases: Sequence[str] = (),
    protect_pos_keywords: Sequence[str] = DEFAULT_PROTECT_POS_KEYWORDS,
) -> list[Range]:
    """品詞と保護語句から変換対象外の範囲を集める。"""
    ranges = [
        (token_span.start, token_span.end)
        for token_span in token_spans
        if is_protected_token(token_span, protect_pos_keywords=protect_pos_keywords)
    ]
    ranges.extend(find_phrase_ranges(text, protected_phrases))
    return merge_ranges(ranges)


def build_protected_ranges(
    text: str,
    *,
    sudachi_tokens: Sequence[Morpheme],
    protected_phrases: Sequence[str] = DEFAULT_PROTECTED_PHRASES,
    protect_pos_keywords: Sequence[str] = DEFAULT_PROTECT_POS_KEYWORDS,
) -> list[Range]:
    """Sudachiの解析結果と保護語句から変換対象外の範囲を求める。"""
    token_spans = build_token_spans(text, sudachi_tokens)
    return collect_protected_ranges(
        text,
        token_spans,
        protected_phrases=protected_phrases,
        protect_pos_keywords=protect_pos_keywords,
    )


def contains_big_unit(text: str) -> bool:
    """万・億などの大数の単位を含むか判定する。"""
    return any(unit in text for unit in BIG_UNITS)


def parse_japanese_number(text: str) -> int | None:
    """漢数字と算用数字が混在した表記を整数に変換する。"""
    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = strip_valid_thousands_commas(normalized)
    if not normalized:
        return None

    if ASCII_INTEGER_RE.fullmatch(normalized):
        return int(normalized)

    if not NUMERIC_SURFACE_RE.fullmatch(normalized):
        return None

    if all(char.isdigit() or char in KANJI_DIGITS for char in normalized):
        digits = [char if char.isdigit() else str(KANJI_DIGITS[char]) for char in normalized]
        return int("".join(digits))

    total = 0
    section = 0
    number: int | None = None

    for char in normalized:
        if char.isdigit():
            number = (0 if number is None else number) * 10 + int(char)
        elif char in KANJI_DIGITS:
            number = KANJI_DIGITS[char]
        elif char in SMALL_UNITS:
            number_for_unit = 1 if number is None else number
            section += number_for_unit * SMALL_UNITS[char]
            number = None
        elif char in BIG_UNITS:
            section += 0 if number is None else number
            if section == 0:
                section = 1
            total += section * BIG_UNITS[char]
            section = 0
            number = None
        else:
            return None

    if number is not None:
        section += number

    return total + section


def format_int_with_japanese_big_units(value: int) -> str:
    """万・億などの単位を使用して整数を表記する。"""
    if value == 0:
        return "0"

    parts: list[str] = []
    remainder = value

    for unit in BIG_UNIT_ORDER:
        base = BIG_UNITS[unit]
        quotient = remainder // base
        if quotient:
            parts.append(f"{quotient}{unit}")
            remainder %= base

    if remainder:
        parts.append(str(remainder))

    return "".join(parts)


def normalize_number_surface_expr(
    expr: str,
    *,
    always_use_japanese_big_units: bool = True,
    big_unit_threshold: int = BIG_UNIT_THRESHOLD,
) -> str | None:
    """数値を、万・億などを使った日本語の表記に正規化する。

    例:
        9999      -> 9999
        10000     -> 1万
        10200     -> 1万200
        100000000 -> 1億
        三億五千万 -> 3億5000万

    """
    if DECIMAL_SURFACE_RE.fullmatch(expr):
        return expr

    value = parse_japanese_number(expr)
    if value is None:
        return None

    if always_use_japanese_big_units and abs(value) >= big_unit_threshold:
        return format_int_with_japanese_big_units(value)

    return str(value)


def normalize_decimal_surface_expr(
    expr: str,
    *,
    always_use_japanese_big_units: bool = True,
    big_unit_threshold: int = BIG_UNIT_THRESHOLD,
) -> str | None:
    """算用数字の小数表記を正規化する。

    小数部分は維持する。整数部分が閾値以上かつ10000の倍数でない場合に
    大数単位への変換を許す。「1万0.5」のような表記は生成しない。
    CER用の呼び出しでは大数単位への変換を無効にする。

    例:
        10000.5        -> 10000.5
        33000.75       -> 3万3000.75
        12345.67       -> 1万2345.67
        100000000.25   -> 100000000.25
        123456789.99   -> 1億2345万6789.99

    """
    normalized = unicodedata.normalize("NFKC", expr).strip()
    normalized = strip_valid_thousands_commas(normalized)

    if not DECIMAL_SURFACE_RE.fullmatch(normalized):
        return None

    integer_part, fractional_part = normalized.split(".", 1)
    if "," in integer_part:
        return None
    integer_value = int(integer_part) if integer_part else 0

    use_big_units = (
        always_use_japanese_big_units
        and integer_value >= big_unit_threshold
        and integer_value % BIG_UNIT_THRESHOLD != 0
    )

    if use_big_units:
        integer_surface = format_int_with_japanese_big_units(integer_value)
    else:
        integer_surface = str(integer_value)

    return f"{integer_surface}.{fractional_part}"


def normalize_numeric_surface_expr(
    expr: str,
    *,
    always_use_japanese_big_units: bool = True,
    big_unit_threshold: int = BIG_UNIT_THRESHOLD,
) -> str | None:
    """整数または小数の数値表記を正規化する。"""
    normalized = unicodedata.normalize("NFKC", expr).strip()
    normalized = strip_valid_thousands_commas(normalized)

    if DECIMAL_SURFACE_RE.fullmatch(normalized):
        return normalize_decimal_surface_expr(
            normalized,
            always_use_japanese_big_units=always_use_japanese_big_units,
            big_unit_threshold=big_unit_threshold,
        )

    return normalize_number_surface_expr(
        normalized,
        always_use_japanese_big_units=always_use_japanese_big_units,
        big_unit_threshold=big_unit_threshold,
    )


def is_decimal_surface_expr(expr: str) -> bool:
    """表記を簡易変換し、算用数字の小数か判定する。"""
    normalized = unicodedata.normalize("NFKC", expr).strip()
    normalized = strip_valid_thousands_commas(normalized)
    return bool(DECIMAL_SURFACE_RE.fullmatch(normalized))


def _is_number_pos(pos: Sequence[str]) -> bool:
    """Sudachiの品詞が数詞か判定する。"""
    return len(pos) >= 2 and pos[0] == "名詞" and pos[1] == "数詞"


def _is_counter_pos(surface: str, pos: Sequence[str], counter_suffixes: Sequence[str]) -> bool:
    """形態素を助数詞として扱えるか判定する。"""
    if surface in counter_suffixes:
        return True
    return any(part in {"助数詞", "助数詞可能"} for part in pos)


def _morpheme_pos(morpheme: Morpheme) -> tuple[str, ...]:
    try:
        return tuple(morpheme.part_of_speech())
    except Exception:
        return ()


def _split_morpheme_to_short_units(morpheme: Morpheme) -> tuple[Morpheme, ...]:
    if sudachi_tokenizer is None:
        return ()
    try:
        return tuple(morpheme.split(sudachi_tokenizer.Tokenizer.SplitMode.A))
    except Exception:
        return ()


def _split_numeric_counter_surface(
    surface: str,
    counter_suffixes: Sequence[str],
) -> tuple[str, str] | None:
    for suffix in _prepare_unique_longest_first(tuple(counter_suffixes)):
        if not suffix or not surface.endswith(suffix):
            continue

        number_surface = surface[: -len(suffix)]
        if not number_surface:
            continue

        if parse_japanese_number(number_surface) is None:
            continue

        return number_surface, suffix

    return None


def _is_lexical_numeric_counter_surface(
    surface: str,
    counter_suffixes: Sequence[str],
) -> bool:
    """Sudachiが1語として扱う、人数・年齢などの数値表現を許容する。"""
    split_surface = _split_numeric_counter_surface(surface, counter_suffixes)
    if split_surface is None:
        return False

    number_surface, suffix = split_surface
    if suffix not in LEXICAL_NUMERIC_COUNTER_SUFFIXES:
        return False

    value = parse_japanese_number(number_surface)
    if value is None:
        return False

    if suffix in {"歳", "歳児", "才"}:
        return 0 <= value <= MAX_LEXICAL_AGE

    return True


def _is_numeric_expression_part(
    surface: str,
    pos: Sequence[str],
    counter_suffixes: Sequence[str],
) -> bool:
    return (
        _is_number_pos(pos)
        or _is_counter_pos(surface, pos, counter_suffixes)
        or _is_lexical_numeric_counter_surface(surface, counter_suffixes)
    )


def _is_numeric_expression_token(morpheme: Morpheme, counter_suffixes: Sequence[str]) -> bool:
    surface = morpheme.surface()
    if not NUMERIC_CANDIDATE_RE.search(surface):
        return True

    pos = _morpheme_pos(morpheme)
    if _is_number_pos(pos) or _is_lexical_numeric_counter_surface(surface, counter_suffixes):
        return True

    split_units = _split_morpheme_to_short_units(morpheme)
    if not split_units:
        return False

    has_number = False
    for unit in split_units:
        unit_surface = unit.surface()
        unit_pos = _morpheme_pos(unit)
        if not _is_numeric_expression_part(unit_surface, unit_pos, counter_suffixes):
            return False
        if _is_number_pos(unit_pos) or _is_lexical_numeric_counter_surface(
            unit_surface, counter_suffixes
        ):
            has_number = True

    return has_number


def build_numeric_candidate_protected_ranges(
    text: str,
    *,
    sudachi_tokens: Sequence[Morpheme],
    counter_suffixes: Sequence[str] = DEFAULT_COUNTER_SUFFIXES,
) -> list[Range]:
    """数詞と解析されない、数字に見える形態素を保護する。

    「万歳」「五月雨」などの一般語の一部分を、正規表現だけで誤変換する
    ことを防ぐ。
    """
    if not text or not NUMERIC_CANDIDATE_RE.search(text):
        return []

    token_spans = build_token_spans(text, sudachi_tokens)
    ranges: list[Range] = []
    for morpheme, token_span in zip(sudachi_tokens, token_spans, strict=False):
        if NUMERIC_CANDIDATE_RE.search(token_span.surface) and not _is_numeric_expression_token(
            morpheme,
            counter_suffixes,
        ):
            ranges.append((token_span.start, token_span.end))

    return merge_ranges(ranges)


@cache
def _compile_numeric_patterns(counter_suffixes: tuple[str, ...]) -> NumericPatterns:
    """助数詞の設定に合わせて数値用の正規表現をコンパイルする。"""
    suffix_union = "|".join(
        re.escape(suffix) for suffix in _prepare_unique_longest_first(counter_suffixes)
    )

    return NumericPatterns(
        date_ymd=re.compile(
            rf"(?<![0-9A-Za-z])({NUM_TOKEN_RE})\s*年\s*({NUM_TOKEN_RE})\s*月\s*({NUM_TOKEN_RE})\s*日(?![0-9A-Za-z])",
        ),
        date_ym=re.compile(
            rf"(?<![0-9A-Za-z])({NUM_TOKEN_RE})\s*年\s*({NUM_TOKEN_RE})\s*月(?![0-9A-Za-z])",
        ),
        date_md=re.compile(
            rf"(?<![0-9A-Za-z])({NUM_TOKEN_RE})\s*月\s*({NUM_TOKEN_RE})\s*日(?![0-9A-Za-z])",
        ),
        quarter=re.compile(
            rf"(?<![0-9A-Za-z])第\s*({NUM_TOKEN_RE})\s*四半期(?![0-9A-Za-z])",
        ),
        score=re.compile(
            rf"(?<![0-9A-Za-z])({NUM_TOKEN_RE})\s*(?:-|対)\s*({NUM_TOKEN_RE})(?![0-9A-Za-z])",
        ),
        range_expr=re.compile(
            rf"(?<![0-9A-Za-z])({NUM_TOKEN_RE})\s*(?:〜|から)\s*({NUM_TOKEN_RE})(?![0-9A-Za-z])",
        ),
        decimal_counter=re.compile(
            rf"(?<![0-9A-Za-z])({NUM_OR_DECIMAL_TOKEN_RE})(?:\s*(から|〜)\s*({NUM_OR_DECIMAL_TOKEN_RE}))?\s*({suffix_union})(?![0-9A-Za-z])",
        ),
        decimal_plain=re.compile(
            rf"(?<![0-9A-Za-z])({DECIMAL_TOKEN_RE})(?![0-9A-Za-z])",
        ),
        counter=re.compile(
            rf"(?<![0-9A-Za-z])({NUM_TOKEN_RE})(?:\s*(から|〜)\s*({NUM_OR_DECIMAL_TOKEN_RE}))?\s*({suffix_union})(?![0-9A-Za-z])",
        ),
        plain=re.compile(
            rf"(?<![0-9A-Za-z])({NUM_TOKEN_RE})(?![0-9A-Za-z])",
        ),
    )


def make_numeric_patterns(counter_suffixes: Sequence[str]) -> NumericPatterns:
    """助数詞の設定に対応する正規表現を取得する。"""
    return _compile_numeric_patterns(tuple(counter_suffixes))


def _stash_placeholder(placeholders: dict[str, str], prefix: str, value: str) -> str:
    """一時置換用の文字列を保存し、そのキーを返す。"""
    key = f"__{prefix}_{len(placeholders)}__"
    placeholders[key] = value
    return key


def _restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    """一時置換した文字列を登録順に復元する。"""
    restored = text
    for key, value in placeholders.items():
        restored = restored.replace(key, value)
    return restored


def normalize_numeric_chunk(
    text: str,
    *,
    counter_suffixes: Sequence[str] = DEFAULT_COUNTER_SUFFIXES,
    normalize_dates: bool = True,
    normalize_score_and_range: bool = True,
    normalize_plain_numbers: bool = False,
    normalize_quarters: bool = True,
    quarter_only_1_to_4: bool = True,
    decimal_always_use_japanese_big_units: bool = True,
    patterns: NumericPatterns | None = None,
) -> str:
    """保護対象外のテキスト内の数値表現を正規化する。"""
    if not text or not NUMERIC_CANDIDATE_RE.search(text):
        return text

    compiled_patterns = (
        patterns if patterns is not None else make_numeric_patterns(counter_suffixes)
    )
    normalized = text

    def convert(value: str) -> str | None:
        return normalize_number_surface_expr(
            value,
            always_use_japanese_big_units=True,
            big_unit_threshold=BIG_UNIT_THRESHOLD,
        )

    def convert_decimal(value: str) -> str | None:
        return normalize_decimal_surface_expr(
            value,
            always_use_japanese_big_units=decimal_always_use_japanese_big_units,
            big_unit_threshold=BIG_UNIT_THRESHOLD,
        )

    def convert_numeric(value: str) -> str | None:
        return normalize_numeric_surface_expr(
            value,
            always_use_japanese_big_units=True,
            big_unit_threshold=BIG_UNIT_THRESHOLD,
        )

    decimal_placeholders: dict[str, str] = {}

    def stash_decimal(match: re.Match[str]) -> str:
        raw = match.group(1)
        value = convert_decimal(raw)
        if value is None:
            return match.group(0)
        return _stash_placeholder(decimal_placeholders, "DECIMAL", value)

    def stash_decimal_counter(match: re.Match[str]) -> str:
        raw = match.group(1)
        separator = match.group(2)
        right_raw = match.group(3)
        suffix = match.group(4)
        if separator is None or right_raw is None:
            if not is_decimal_surface_expr(raw):
                return match.group(0)

            value = convert_decimal(raw)
            if value is None:
                return match.group(0)

            return _stash_placeholder(decimal_placeholders, "DECIMAL", f"{value}{suffix}")

        if not (is_decimal_surface_expr(raw) or is_decimal_surface_expr(right_raw)):
            return match.group(0)

        value = convert_numeric(raw)
        if value is None:
            return match.group(0)

        right_value = convert_numeric(right_raw)
        if right_value is None:
            return match.group(0)

        return _stash_placeholder(
            decimal_placeholders, "DECIMAL", f"{value}{separator}{right_value}{suffix}"
        )

    normalized = compiled_patterns.decimal_counter.sub(stash_decimal_counter, normalized)
    normalized = compiled_patterns.decimal_plain.sub(stash_decimal, normalized)

    if normalize_dates:

        def repl_date_ymd(match: re.Match[str]) -> str:
            year = convert(match.group(1))
            month = convert(match.group(2))
            day = convert(match.group(3))
            if year is None or month is None or day is None:
                return match.group(0)
            return f"{year}年{month}月{day}日"

        def repl_date_ym(match: re.Match[str]) -> str:
            year = convert(match.group(1))
            month = convert(match.group(2))
            if year is None or month is None:
                return match.group(0)
            return f"{year}年{month}月"

        def repl_date_md(match: re.Match[str]) -> str:
            month = convert(match.group(1))
            day = convert(match.group(2))
            if month is None or day is None:
                return match.group(0)
            return f"{month}月{day}日"

        normalized = compiled_patterns.date_ymd.sub(repl_date_ymd, normalized)
        normalized = compiled_patterns.date_ym.sub(repl_date_ym, normalized)
        normalized = compiled_patterns.date_md.sub(repl_date_md, normalized)

    quarter_placeholders: dict[str, str] = {}
    if normalize_quarters:

        def normalize_quarter_num(raw_num: str) -> str | None:
            quarter = convert(raw_num)
            if quarter is None:
                return None

            try:
                quarter_int = int(quarter)
            except ValueError:
                return None

            if quarter_only_1_to_4 and quarter_int not in {1, 2, 3, 4}:
                return None

            return str(quarter_int)

        def repl_quarter(match: re.Match[str]) -> str:
            quarter = normalize_quarter_num(match.group(1))
            if quarter is None:
                return match.group(0)
            return f"第{quarter}四半期"

        def stash_quarter(match: re.Match[str]) -> str:
            return _stash_placeholder(quarter_placeholders, "QUARTER", match.group(0))

        normalized = compiled_patterns.quarter.sub(repl_quarter, normalized)
        normalized = NORMALIZED_QUARTER_RE.sub(stash_quarter, normalized)

    if normalize_score_and_range:

        def repl_score(match: re.Match[str]) -> str:
            left = convert(match.group(1))
            right = convert(match.group(2))
            if left is None or right is None:
                return match.group(0)
            return f"{left}対{right}"

        def repl_range(match: re.Match[str]) -> str:
            left = convert(match.group(1))
            right = convert(match.group(2))
            if left is None or right is None:
                return match.group(0)
            return f"{left}から{right}"

        normalized = compiled_patterns.range_expr.sub(repl_range, normalized)
        normalized = compiled_patterns.score.sub(repl_score, normalized)

    def repl_counter(match: re.Match[str]) -> str:
        number = convert(match.group(1))
        separator = match.group(2)
        right_raw = match.group(3)
        suffix = match.group(4)
        if number is None:
            return match.group(0)
        if separator is None or right_raw is None:
            return f"{number}{suffix}"

        right_value = convert_numeric(right_raw)
        if right_value is None:
            return match.group(0)

        return f"{number}{separator}{right_value}{suffix}"

    normalized = compiled_patterns.counter.sub(repl_counter, normalized)

    if normalize_plain_numbers:

        def repl_plain(match: re.Match[str]) -> str:
            number = convert(match.group(1))
            return number if number is not None else match.group(0)

        normalized = compiled_patterns.plain.sub(repl_plain, normalized)

    normalized = _restore_placeholders(normalized, quarter_placeholders)
    return _restore_placeholders(normalized, decimal_placeholders)


def normalize_japanese_numeric_surface(
    text: str,
    *,
    sudachi_tokens: Sequence[Morpheme],
    protected_ranges: Sequence[Range] = (),
    protected_phrases: Sequence[str] = DEFAULT_PROTECTED_PHRASES,
    protect_pos_keywords: Sequence[str] = DEFAULT_PROTECT_POS_KEYWORDS,
    counter_suffixes: Sequence[str] = DEFAULT_COUNTER_SUFFIXES,
    normalize_dates: bool = True,
    normalize_quarters: bool = True,
    quarter_only_1_to_4: bool = True,
    normalize_score_and_range: bool = False,
    normalize_plain_numbers: bool = True,
    decimal_always_use_japanese_big_units: bool = True,
) -> str:
    """保護対象の語句を保持し、それ以外の数値表現を正規化する。"""
    if not text or not NUMERIC_CANDIDATE_RE.search(text):
        return text

    protected_ranges = merge_ranges(
        [
            *protected_ranges,
            *build_protected_ranges(
                text,
                sudachi_tokens=sudachi_tokens,
                protected_phrases=protected_phrases,
                protect_pos_keywords=protect_pos_keywords,
            ),
            *build_numeric_candidate_protected_ranges(
                text,
                sudachi_tokens=sudachi_tokens,
                counter_suffixes=counter_suffixes,
            ),
        ],
    )
    compiled_patterns = make_numeric_patterns(counter_suffixes)

    def normalize_chunk(chunk: str) -> str:
        return normalize_numeric_chunk(
            chunk,
            counter_suffixes=counter_suffixes,
            normalize_dates=normalize_dates,
            normalize_quarters=normalize_quarters,
            quarter_only_1_to_4=quarter_only_1_to_4,
            normalize_score_and_range=normalize_score_and_range,
            normalize_plain_numbers=normalize_plain_numbers,
            decimal_always_use_japanese_big_units=decimal_always_use_japanese_big_units,
            patterns=compiled_patterns,
        )

    return apply_to_unprotected_ranges(text, protected_ranges, normalize_chunk)
