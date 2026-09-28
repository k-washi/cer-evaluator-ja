"""正規化後の全文候補から、CERが厳密に最小となる正解を選択する。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from .annotations import reference_candidates
from .normalization import normalize


@dataclass(frozen=True)
class CERResult:
    """採用した正解・正規化後の文字列・編集内訳。cerは百分率ではなく比率。"""

    cer: float
    errors: int
    reference_length: int
    substitutions: int
    deletions: int
    insertions: int
    selected_reference: str
    normalized_reference: str
    normalized_hypothesis: str


@dataclass(frozen=True)
class CorpusResult:
    """各発話のCERを最小化してから編集数と正解文字数を合算した結果。"""

    cer: float
    errors: int
    reference_length: int
    results: tuple[CERResult, ...]


def evaluate(reference: str, hypothesis: str, *, max_candidates: int | None = 100_000) -> CERResult:
    """候補ごとの「編集数 / max(1, 正解文字数)」を最小化する。

    referenceだけを注釈として解釈し、hypothesisは通常の文字列として扱う。
    候補を全文に連結してから正規化するので、タグ境界をまたぐ数値や
    保護語句にも対応する。同率なら編集数が少ない候補、次に記載順を優先。

    RapidFuzzのC++実装で編集距離を求め、暫定最良CERを超える探索は
    score_cutoffで打ち切る。比率の比較には整数を使い丸め誤差を避ける。
    """
    hypothesis = normalize(hypothesis)
    best_errors: int | None = None
    best_length = 1
    selected = normalized = ""
    for candidate in reference_candidates(reference, max_candidates=max_candidates):
        current = normalize(candidate)
        length = max(1, len(current))
        cutoff = None if best_errors is None else best_errors * length // best_length
        # 文字数差だけで現在の最良CERを超える候補は、距離計算も省く。
        if cutoff is not None and abs(len(current) - len(hypothesis)) > cutoff:
            continue
        errors = Levenshtein.distance(current, hypothesis, score_cutoff=cutoff)
        if cutoff is not None and errors > cutoff:
            continue
        comparison = 0 if best_errors is None else errors * best_length - best_errors * length
        if best_errors is None or comparison < 0 or (comparison == 0 and errors < best_errors):
            best_errors, best_length = errors, length
            selected, normalized = candidate, current
            # CER 0より良い候補は存在しない。以降の正規化も不要。
            if errors == 0:
                break
    assert best_errors is not None
    operations = Levenshtein.editops(normalized, hypothesis)
    return CERResult(
        cer=best_errors / best_length,
        errors=best_errors,
        reference_length=len(normalized),
        substitutions=sum(op.tag == "replace" for op in operations),
        deletions=sum(op.tag == "delete" for op in operations),
        insertions=sum(op.tag == "insert" for op in operations),
        selected_reference=selected,
        normalized_reference=normalized,
        normalized_hypothesis=hypothesis,
    )


def cer(reference: str, hypothesis: str, *, max_candidates: int | None = 100_000) -> float:
    """最小の正規化CERを返す。戻り値0.1は10%を表す。"""
    return evaluate(reference, hypothesis, max_candidates=max_candidates).cer


def evaluate_many(
    pairs: Iterable[tuple[str, str]], *, max_candidates: int | None = 100_000
) -> CorpusResult:
    """対応済みの正解・認識結果の組を評価し、編集数と正解文字数を合算する。

    空の正解への挿入も分子に加える。全正解が空なら分母は1とする。
    入力が0件の場合はCER 0。発話別CERの単純平均ではない。
    """
    results = tuple(evaluate(ref, hyp, max_candidates=max_candidates) for ref, hyp in pairs)
    errors = sum(result.errors for result in results)
    length = sum(result.reference_length for result in results)
    return CorpusResult(errors / max(1, length), errors, length, results)
