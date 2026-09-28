"""メールアドレスを正規化の途中で退避し、内部の記号や数字を保護する。"""

from __future__ import annotations

import re

_ATOM = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
_LOCAL = rf'(?:{_ATOM}(?:\.{_ATOM})*|"(?:[^"\\\r\n]|\\[^\r\n])+")'
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_DOMAIN = rf"(?:{_LABEL}(?:\.{_LABEL})+|\[(?:IPv6:)?[0-9A-Fa-f:.]+\])"
_EMAIL = re.compile(
    rf"(?<![A-Za-z0-9_.@]){_LOCAL}@{_DOMAIN}(?![A-Za-z0-9_@-])",
    re.IGNORECASE,
)


def protect_emails(text: str) -> tuple[str, dict[str, str]]:
    """NFKC後のメール表記を退避し、復元用の対応表とともに返す。

    ASCIIのローカル部（引用符付きも含む）とドメイン/IPリテラルを扱う。
    実在性を検証する処理ではない。メールにも英字の小文字化は適用する。
    記号削除だけでなく、アドレス内の数字や空白の変換も避ける。
    """
    if "@" not in text:
        return text, {}
    replacements: dict[str, str] = {}
    original = text.lower()
    marker = "cermailtoken"

    def replace(match: re.Match[str]) -> str:
        nonlocal marker
        # 数字・記号を使わず、原文とも他のマーカーとも衝突しないキーにする。
        while marker in original or marker in replacements:
            marker = marker.removesuffix("token") + "xtoken"
        address = match.group()
        # 外側を一対の単一引用符で囲んだ表記は引用として扱う。
        # o'connor のようなローカル部の途中のアポストロフィは保持する。
        if address.startswith("'") and text[match.end() : match.end() + 1] == "'":
            address = address[1:]
        replacements[marker] = address.lower()
        return marker

    return _EMAIL.sub(replace, text), replacements
