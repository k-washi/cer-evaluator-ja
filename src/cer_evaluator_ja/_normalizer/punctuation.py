"""CERで無視する引用符・中黒・句読点だけを削除する。"""

# 削除対象を限定する。括弧・演算子・ハイフン・下線などはこの集合に含めない。
_IGNORED = frozenset("\"'「」『』“”‘’«»‹›〝〞〟・·‧、。，,!?")


def remove_ignored_punctuation(text: str) -> str:
    """引用符・中黒・句読点・!?を除き、小数点とその他の記号を保持する。

    NFKC適用後の文字列を受け取り、半角数字に挟まれたピリオドは残す。
    メールアドレスは呼び出し元で退避しておき、この処理の後に復元する。
    数値・範囲の解釈が終わってから適用する。
    """
    result = []
    for index, char in enumerate(text):
        if char in _IGNORED:
            continue
        if char == "." and not (
            0 < index < len(text) - 1
            and "0" <= text[index - 1] <= "9"
            and "0" <= text[index + 1] <= "9"
        ):
            continue
        result.append(char)
    return "".join(result).strip()
