"""候補選択・正規化境界・編集内訳を独立した計算と照合する。"""

from fractions import Fraction
from itertools import product
from random import Random

import pytest

from cer_evaluator_ja import (
    CandidateLimitError,
    cer,
    evaluate,
    evaluate_many,
    normalize,
    reference_candidates,
)


def distance(reference, hypothesis):
    """検証用の単純なLevenshtein距離。製品側のRapidFuzzに依存しない。"""
    row = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        next_row = [i]
        for j, right in enumerate(hypothesis, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (left != right)))
        row = next_row
    return row[-1]


@pytest.mark.parametrize("tag", ["D", "F", "VF", "VD"])
def test_optional_tags(tag):
    """言い淀み・フィラーは、認識された場合も省略された場合も許す。"""
    body = "えー|ええ" if tag in {"VF", "VD"} else "えー"
    reference = f"今日は({tag} {body})晴れ"
    assert cer(reference, "今日は晴れ") == 0
    assert cer(reference, "今日はえー晴れ") == 0
    if tag in {"VF", "VD"}:
        assert cer(reference, "今日はええ晴れ") == 0


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("(A はい|)晴れ(A です|ですよ|)", "晴れですよ"),
        ("(A はい|)晴れ(A です|ですよ|)", "晴れ"),
        ("(V iPhone|アイフォーン)", "アイフォーン"),
        ("(L)晴(B)れ(C)", "晴れ"),
        ("（VF まあ｜まぁ）晴れ", "晴れ"),
        ("一(V 万|億)円", "10000円"),
        ("(V 万|千)歳", "万歳"),
        ("(V Open|Closed)AI API", "OpenAI API"),
        ("あー(A ー|)", "あー"),
        ("今日は(A 晴れ|雨).", "今日は晴れ"),
        ("(VF まあ|まぁ)(V iPhone|アイフォーン)は1万5000円(A です|ですよ|)", "iPhoneは15000円"),
    ],
)
def test_annotations_and_whole_text_normalization(reference, hypothesis):
    """タグの境界をまたぐ数字・保護語・英語・長音も全文で正規化する。"""
    assert cer(reference, hypothesis) == 0


def test_minimize_ratio_not_distance():
    """距離1の短い候補より、距離2でもCERが低い長い候補を選ぶ。"""
    result = evaluate("(V あ|あいうえ)", "あい")
    assert result.selected_reference == "あいうえ"
    assert result.errors == 2
    assert result.cer == 0.5


def test_equal_ratio_prefers_fewer_errors_then_order():
    """同率では編集数、さらに同じなら注釈の記載順で決定する。"""
    assert evaluate("(V ああ|あ)", "い").selected_reference == "あ"
    assert evaluate("(V あ|う)", "い").selected_reference == "あ"


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("", "", 0),
        ("(L)(B)(C)", "", 0),
        ("", "あい", 2),
        ("(F えー)", "", 0),
        ("(A はい|)", "", 0),
        ("あ", "", 1),
    ],
)
def test_empty_reference_convention(reference, hypothesis, expected):
    """正解0文字の分母は1とし、認識結果はすべて挿入とする。"""
    assert cer(reference, hypothesis) == expected


@pytest.mark.parametrize(
    ("reference", "hypothesis", "counts"),
    [("あいう", "あえう", (1, 0, 0)), ("あいう", "あう", (0, 1, 0)), ("あう", "あいう", (0, 0, 1))],
)
def test_edit_counts(reference, hypothesis, counts):
    """置換・削除・挿入の向きと合計を検証する。"""
    result = evaluate(reference, hypothesis)
    assert (result.substitutions, result.deletions, result.insertions) == counts
    assert result.errors == sum(counts)


def test_randomized_against_independent_exhaustive_oracle():
    """可変長候補の厳密解と照合し、打ち切りによる取りこぼしを検出する。"""
    rng = Random(2026)
    vocabulary = ["", "あ", "い", "あいう", "うううう", "えあ"]
    for _ in range(200):
        groups = [rng.sample(vocabulary, 3) for _ in range(rng.randint(1, 4))]
        reference = "え" + "".join(f"(A {'|'.join(group)})" for group in groups)
        hypothesis = "".join(rng.choices("あいうえ", k=rng.randrange(12)))
        candidates = ["え" + "".join(parts) for parts in product(*groups)]
        best = min(
            candidates,
            key=lambda text: (
                Fraction(distance(text, hypothesis), len(text)),
                distance(text, hypothesis),
            ),
        )
        result = evaluate(reference, hypothesis)
        assert result.normalized_reference == best
        assert result.errors == distance(best, hypothesis)
        assert result.cer == distance(best, hypothesis) / len(best)


def test_corpus_micro_average_and_empty_insertions():
    """発話別最適化のあと文字数で重み付けし、無音区間の誤挿入も数える。"""
    result = evaluate_many([("あ", "い"), ("あいうえ", "あいうえ"), ("", "う")])
    assert result.cer == 2 / 5
    assert result.errors == 2
    assert result.reference_length == 5
    assert evaluate_many([]).cer == 0
    assert evaluate_many([("", "あい")]).cer == 2


def test_candidate_limit_is_explicit_not_approximation():
    """探索上限を超えた場合は不完全なCERを返さない。"""
    with pytest.raises(CandidateLimitError):
        evaluate("(F あ)" * 20, "", max_candidates=100)
    assert cer("(F あ)" * 5, "", max_candidates=None) == 0
    with pytest.raises(ValueError):
        list(reference_candidates("あ", max_candidates=0))


@pytest.mark.parametrize(
    "reference",
    ["(X あ)", "(V あ", "(V あ|(F え))", "(F)", "(V あ|)", "(A |)", "(F あ|え)", "(L 笑)"],
)
def test_invalid_annotations(reference):
    """不明・不正なタグを本文として黙って採点しない。"""
    with pytest.raises(ValueError):
        evaluate(reference, "あ")


def test_plain_parentheses_and_prediction_are_literal():
    """通常の括弧を保持し、認識結果の注釈を評価上有利に解釈しない。"""
    assert cer("声(小声)", "声(小声)") == 0
    assert cer("あ", "(F あ)") > 0
    assert list(reference_candidates("(A はい|   )")) == ["はい", ""]
    assert normalize("(F あ)") != "あ"
