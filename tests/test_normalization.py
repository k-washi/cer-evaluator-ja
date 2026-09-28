"""基本正規化とCER向け追加規則の最終出力を回帰検証する。"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from cer_evaluator_ja import normalize
from cer_evaluator_ja.normalization import _normalize_prepared


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", ""),
        ("  私 は  「 こんにちは ！ 」 と 言った 。  ", "私はこんにちはと言った"),
        ("\u3000ｶﾞｯﾂﾎﾟｰｽﾞ", "ガッツポーズ"),
        ("COVID - 19", "covid-19"),
        ("ASR / TTS", "asr/tts"),
        ("36 ℃", "36°c"),
        ("¥ 1000", "¥1000"),
        ("料金 は ¥1000", "料金は¥1000"),
        ("イェーーーーい！", "イェーい"),
        ("二千二十四年三月一日に発売", "2024年3月1日に発売"),
        ("一億二千年前", "1億2000年前"),
        ("一万二百円かかる", "1万200円かかる"),
        ("3億5,000万の予算", "3億5000万の予算"),
        ("100,242", "10万242"),
        ("100,242円", "10万242円"),
        ("10737,418", "10737418"),
        ("10,737,418", "1073万7418"),
        ("10,10", "1010"),
        ("1,2,3", "123"),
        ("第3四半期の決算", "第3四半期の決算"),
        ("十三円", "13円"),
        ("十-三で勝った", "10-3で勝った"),
        ("OpenAI APIを使う", "openai apiを使う"),
        ("GPUメモリが足りない", "gpuメモリが足りない"),
        ("ＯｐｅｎＡＩ APIを使う", "openai apiを使う"),
        ("PyTorchで、学習する。", "pytorchで学習する"),
        ("はい.", "はい"),
        ("はい .", "はい"),
        ("はい．", "はい"),
        ("今日は晴れです.明日も晴れです", "今日は晴れです明日も晴れです"),
        ("第3四半期の決算です", "第3四半期の決算です"),
        ("第5四半期の決算です", "第5四半期の決算です"),
        ("一〜三話を見る", "1〜3話を見る"),
        ("使う AIは\u3000\u3000ChatGPT です！", "使うaiはchatgptです"),
        ("10000.5", "10000.5"),
        ("1.2億円", "1.2億円"),
        ("33,000.75ドル", "33000.75ドル"),
        ("10737,418ドル", "10737418ドル"),
        ("10,737,418ドル", "1073万7418ドル"),
        ("10737,418.5ドル", "10737418.5ドル"),
        ("10,737,418.5ドル", "10737418.5ドル"),
        ("10737418.5ドル", "10737418.5ドル"),
        ("30000.75ドル", "30000.75ドル"),
        ("0.5畳のスペース", "0.5畳のスペース"),
        ("3.", "3"),
        ("一番大きい数字は 100,000,000,000 です", "1番大きい数字は1000億です"),
        ("第五人格を遊ぶ", "第五人格を遊ぶ"),
        ("四月は君の嘘を見た", "四月は君の嘘を見た"),
        ("十角館へ行く", "十角館へ行く"),
        ("六義園を散歩する", "六義園を散歩する"),
        ("つぎの瞬間万歳万歳の声が聞こえていた。", "つぎの瞬間万歳万歳の声が聞こえていた"),
        ("万引きは犯罪です。", "万引きは犯罪です"),
        ("五月雨式で進める。", "五月雨式で進める"),
        ("七夕祭りに行く。", "七夕祭りに行く"),
        ("四季を感じる。", "四季を感じる"),
        ("二次元コードを読む。", "二次元コードを読む"),
        ("四半期の決算です。", "四半期の決算です"),
        ("億ションを買う。", "億ションを買う"),
        ("万年筆で書く。", "万年筆で書く"),
        ("千鳥足になる。", "千鳥足になる"),
        ("十字架を見る。", "十字架を見る"),
        ("一式そろえる。", "一式そろえる"),
        ("五輪大会を見る。", "五輪大会を見る"),
        ("第三者の意見です。", "第三者の意見です"),
        ("一万円かかる。", "1万円かかる"),
        ("十三円です。", "13円です"),
        ("三人で五千円です。", "3人で5000円です"),
        ("一回だけです。", "1回だけです"),
        ("二人組です。", "2人組です"),
        ("二十歳です。", "20歳です"),
        ("第三四半期の決算です。", "第3四半期の決算です"),
        ("第十五回です。", "第15回です"),
        ("四月十三日です。", "4月13日です"),
        ("一〜三話を見る。", "1〜3話を見る"),
    ],
)
def test_reference_normalization(source, expected):
    """基本正規化のあと、指定した引用符・句読点等だけを除いた結果を検証する。"""
    assert normalize(source) == expected


def test_thread_local_tokenizers():
    """複数スレッドからの数値正規化が同じ結果になる。"""
    texts = [f"{i}万円" for i in range(10, 30)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(normalize, texts)) == texts


def test_no_tokenization_without_numeric_characters(monkeypatch):
    """数値を含まない候補では辞書の生成も形態素解析も不要。"""
    from threading import local

    from cer_evaluator_ja import normalization

    def fail(**kwargs):
        raise AssertionError("不要な形態素解析")

    _normalize_prepared.cache_clear()
    monkeypatch.setattr(normalization, "_STATE", local())
    monkeypatch.setattr(normalization, "Dictionary", fail)
    assert normalize("　Ｈｅｌｌｏ　こんにちはーー！") == "helloこんにちはー"


@pytest.mark.parametrize(
    ("source", "equivalent", "expected"),
    [
        ("ゼロ", "0", "0"),
        ("ｾﾞﾛ", "０", "0"),
        ("819ゼロ5", "81905", "8万1905"),
        ("ゼロ円", "0円", "0円"),
        ("03-1234-5678", "0312345678", "0312345678"),
        ("０９０－１２３４－５６７８", "09012345678", "09012345678"),
        ("0120-123-456", "0120123456", "0120123456"),
        ("電話は03-1234-5678です", "電話は0312345678です", "電話は0312345678です"),
        ("+81-90-1234-5678", "+819012345678", "+819012345678"),
        ("連絡先は+1-212-555-0123です", "連絡先は+12125550123です", "連絡先は+12125550123です"),
        ("123-4567", "1234567", "123万4567"),
        ("012-3456", "0123456", "0123456"),
        ("〒123-4567", "〒1234567", "〒123万4567"),
        ("1,234,567円", "1234567円", "123万4567円"),
        ("3人から5人まで来ます", "3人〜5人来ます", "3人〜5人来ます"),
        ("10時から12時です", "10時〜12時です", "10時〜12時です"),
        ("三人から五人まで", "3人〜5人", "3人〜5人"),
        ("1.5から2.5まで", "1.5〜2.5", "1.5〜2.5"),
        ("3～5まで", "3~5", "3〜5"),
        ("3センチメートル", "3cm", "3cm"),
        ("3センチ", "3シーエム", "3cm"),
        ("三シーエム", "3cm", "3cm"),
        ("5ミリメートル", "5mm", "5mm"),
        ("5ミリ", "5エムエム", "5mm"),
        ("３センチ", "3CM", "3cm"),
        ("3センチから5センチまで", "3cm〜5cm", "3cm〜5cm"),
        ("1.5ミリメートルから2.5ミリまで", "1.5mm〜2.5mm", "1.5mm〜2.5mm"),
        ("GPU", "gpu", "gpu"),
        ("iPhone", "iphone", "iphone"),
    ],
)
def test_additional_numeric_surfaces(source, equivalent, expected):
    """表記の違う原文同士が同じ比較用文字列になる。"""
    from cer_evaluator_ja import cer

    assert normalize(source) == expected
    assert normalize(equivalent) == expected
    assert cer(source, equivalent) == 0
    assert cer(equivalent, source) == 0


@pytest.mark.parametrize(
    "source",
    [
        "イチ",
        "いち",
        "イチゴ",
        "れい",
        "4/13",
        "1/2",
        "3-2",
        "03-04",
        "090-1234-5678-9",
        "-3から5",
        "3から-5",
        "春から秋まで",
        "センチメンタル",
    ],
)
def test_additional_rules_do_not_rewrite_other_surfaces(source):
    """読み・分数・短いハイフン表記・数値以外の範囲は変換しない。"""
    # 元からある数値規則（先頭0の除去など）の影響を除いて追加規則を確認する。
    from cer_evaluator_ja._normalizer.surface import (
        normalize_numeric_ranges,
        prepare_numeric_surfaces,
    )

    assert prepare_numeric_surfaces(source) == source
    assert normalize_numeric_ranges(source) == source


def test_no_reading_generation():
    """イチや英語普通名詞を読み・数字へ寄せない。"""
    from cer_evaluator_ja import cer

    assert normalize("イチ") == "イチ"
    assert cer("イチ", "1") > 0
    assert normalize("web server") == "web server"
    assert cer("server", "サーバー") > 0


def test_annotations_can_cross_additional_normalization_boundaries():
    """全文への候補展開のあとで、ゼロ・単位・範囲・番号を正規化する。"""
    from cer_evaluator_ja import cer

    assert cer("ゼ(A ロ|)", "0") == 0
    assert cer("3(V センチ|cm)から5(V センチ|cm)まで", "3cm〜5cm") == 0
    assert cer("03(A -|)1234(A -|)5678", "0312345678") == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("「こんにちは」・ありがとう。", "こんにちはありがとう"),
        ("『こんにちは』【世界】（テスト）“引用”「文」", "こんにちは【世界】(テスト)引用文"),
        ("Hello, world!?", "hello world"),
        ("Hello.", "hello"),
        ("今日は晴れです.明日も晴れです", "今日は晴れです明日も晴れです"),
        ("3.", "3"),
        ("3．", "3"),
        ("3.14。", "3.14"),
        ("３．１４！", "3.14"),
        ("0.5です？", "0.5です"),
        ("12.50", "12.50"),
        ("1..2", "12"),
        ("★GPU★", "★gpu★"),
        ("A+B=C", "a+b=c"),
        ("。！？「」・…", ""),
        ("コーヒー", "コーヒー"),
        ("123,456.78", "123456.78"),
    ],
)
def test_remove_only_ignored_punctuation(source, expected):
    """引用符・中黒・文末のピリオドは除き、数値の小数点と長音は保持する。"""
    assert normalize(source) == expected


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("「東京」・大阪。", "東京大阪"),
        ("こんにちは！？", "こんにちは"),
        ("Hello.", "hello"),
        ("今日は晴れ.明日も晴れ", "今日は晴れ明日も晴れ"),
        ("１２．５！", "12.5"),
        ("(V 東京|とうきょう)・大阪(L)", "東京大阪"),
        ("！？", ""),
    ],
)
def test_cer_ignores_symbols(reference, hypothesis):
    """正解・認識結果の双方で記号を除き、比較文字数にも含めない。"""
    from cer_evaluator_ja import evaluate

    result = evaluate(reference, hypothesis)
    assert result.cer == 0
    assert result.reference_length == len(result.normalized_reference)
    assert result.reference_length == len(normalize(hypothesis))


def test_decimal_point_remains_significant():
    """小数点を落とした認識結果は、記号無視でも正解にはしない。"""
    from cer_evaluator_ja import cer

    assert cer("1.5", "15") > 0
    assert cer("0.5", "05") > 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("１＋２＝３！", "1+2=3"),
        ("3−2=1", "3−2=1"),
        ("3-2=1", "3-2=1"),
        ("x_y", "x_y"),
        ("2*(3+4)=14", "2*(3+4)=14"),
        ("[a+b]/{c_d}", "[a+b]/{c_d}"),
        ("a×b÷c≤d", "a×b÷c≤d"),
        ("x^2", "x^2"),
        ("file-name.txt", "file-nametxt"),
        ("Hello: world;", "hello:world;"),
        ("3から5まで", "3〜5"),
        ("５０％", "50%"),
    ],
)
def test_preserve_math_and_other_symbols(source, expected):
    """削除対象外の演算子・括弧・ハイフン・下線などを残す。"""
    assert normalize(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Foo.Bar@example.com", "foo.bar@example.com"),
        ("a.b+tag!?@example-domain.com!", "a.b+tag!?@example-domain.com"),
        ("連絡はfoo!?@example.com。", "連絡はfoo!?@example.com"),
        ("「Foo.Bar@example.com」へ。", "foo.bar@example.comへ"),
        ("'a.b@example.com'", "a.b@example.com"),
        ('"a,b!?"@example.com!', '"a,b!?"@example.com'),
        ('"a b"@example.com', '"a b"@example.com'),
        ("a_b-c+d=e@example.co.jp", "a_b-c+d=e@example.co.jp"),
        ("o'connor@example.com", "o'connor@example.com"),
        ("!#$%&'*+/=?^_`{|}~-@example.com", "!#$%&'*+/=?^_`{|}~-@example.com"),
        ("100000+03-1234-5678@example.com", "100000+03-1234-5678@example.com"),
        ("a@12345.example.com", "a@12345.example.com"),
        ("a@[127.0.0.1]", "a@[127.0.0.1]"),
        ("a@[IPv6:2001:db8::1]", "a@[ipv6:2001:db8::1]"),
        ("Ａ．Ｂ！＠ＥＸＡＭＰＬＥ．ＣＯＭ", "a.b!@example.com"),
        ("a.b@example.com c.d!?@example.org", "a.b@example.com c.d!?@example.org"),
        ("cermailtoken a!@b.com", "cermailtoken a!@b.com"),
        (
            "cermailtoken cermailxtoken a!@b.com c?@d.com",
            "cermailtoken cermailxtoken a!@b.com c?@d.com",
        ),
        ("foo@example.com! 次です。", "foo@example.com 次です"),
    ],
)
def test_email_is_protected_through_all_normalization(source, expected):
    """メール内の記号・数字・引用ローカル部の空白を保護し、文末記号は除く。"""
    assert normalize(source) == expected


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("a+b=c", "abc"),
        ("3−2", "32"),
        ("a-b", "ab"),
        ("a_b", "ab"),
        ("(a+b)", "a+b"),
        ("a!@example.com", "a@example.com"),
        ("a?@example.com", "a@example.com"),
        ("a.b@example.com", "ab@example.com"),
        ("a+b@example.com", "ab@example.com"),
        ("a@ex-ample.com", "a@example.com"),
    ],
)
def test_significant_symbols_count_as_errors(reference, hypothesis):
    """数式やメールの記号を落とした認識結果は誤りに数える。"""
    from cer_evaluator_ja import cer

    assert cer(reference, hypothesis) > 0


def test_email_and_annotations():
    """候補展開後のメールも保護し、本文の句読点は除く。"""
    from cer_evaluator_ja import cer

    assert (
        cer("連絡先は(V A.B!?@example.com|C.D@example.com)です！", "連絡先はa.b!?@example.comです")
        == 0
    )
