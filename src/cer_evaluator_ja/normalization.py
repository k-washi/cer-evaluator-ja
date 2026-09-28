"""基本のTextNormalizerCERと、数値周辺の追加規則によるCER用正規化。"""

from __future__ import annotations

from functools import lru_cache
from threading import local
from unicodedata import normalize as unicode_normalize

from sudachipy import Dictionary, tokenizer

from ._normalizer.common import normalize_long_vowel_repeats_chunk
from ._normalizer.digit import NUMERIC_CANDIDATE_RE, normalize_japanese_numeric_surface
from ._normalizer.email import protect_emails
from ._normalizer.punctuation import remove_ignored_punctuation
from ._normalizer.spaces import normalize_ja_simple
from ._normalizer.surface import (
    identifier_ranges,
    normalize_numeric_ranges,
    prepare_numeric_surfaces,
)

_STATE = local()


def normalize(text: str) -> str:
    """注釈を含まない全文を正規化する。数字・固有名詞は文脈を考慮する。

    全角半角・空白・長音・ゼロ・番号の区切り・単位を統一した後、
    数値・英字の小文字化・数値範囲を処理し、指定した句読点等だけを除く。
    メール内の記号・数字は保護し、大小文字と文字幅だけを統一する。
    英単語間の空白を維持し、イチの数値化や読み候補の生成は行わない。
    戻り値は比較用。記号削除で数字が隣接し得るため、原文に1回適用する。
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    replacements: dict[str, str] = {}
    if "@" in text or "＠" in text:
        text, replacements = protect_emails(unicode_normalize("NFKC", text))
    text = normalize_ja_simple(text)
    text = normalize_long_vowel_repeats_chunk(text)
    result = _normalize_prepared(prepare_numeric_surfaces(text))
    for marker, email in replacements.items():
        result = result.replace(marker, email)
    return result


@lru_cache(maxsize=4096)
def _normalize_prepared(text: str) -> str:
    """表記統一後の結果を共有し、必要な場合だけ形態素解析する。

    数字候補がなければ数値正規化は恒等変換なのでSudachiを省略できる。
    辞書はスレッドごとに遅延初期化し、キャッシュは4096件を上限とする。
    """
    if NUMERIC_CANDIDATE_RE.search(text):
        if not hasattr(_STATE, "tokenizer"):
            _STATE.tokenizer = Dictionary(dict="core").create()
        tokens = tuple(_STATE.tokenizer.tokenize(text, mode=tokenizer.Tokenizer.SplitMode.C))
        text = normalize_japanese_numeric_surface(
            text,
            sudachi_tokens=tokens,
            protected_ranges=identifier_ranges(text),
            decimal_always_use_japanese_big_units=False,
        )
    return remove_ignored_punctuation(normalize_numeric_ranges(text.lower()))
