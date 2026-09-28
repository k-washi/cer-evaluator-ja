# cer-evaluator-ja

日本語ASRの正解テキストと認識結果から、正規化後のCERを計算します。フィラーの省略や複数の正解候補を許容し、CERが最小になる候補を選びます。Python 3.10以上。GPUは不要です。

## インストール

リポジトリ直下で実行します。

```bash
# pipを使う場合
pip install .

# uvを使う場合
uv sync
```

別のuvプロジェクトに追加する場合は `uv add /path/to/cer-evaluator-ja`。

## Pythonで使う

```python
from cer_evaluator_ja import cer, evaluate

reference = "(F えー)(V iPhone|アイフォーン)は一万五千円(A です|)(L)"
hypothesis = "iPhoneは15000円"

print(cer(reference, hypothesis))  # 0.0（0.1なら10%）

# 採用した正解や誤りの内訳も取得できます
result = evaluate(reference, hypothesis)
print(result.normalized_reference)  # iphoneは1万5000円
print(result.substitutions, result.deletions, result.insertions)  # 0 0 0
```

正解・認識結果は**事前に正規化せず、そのまま渡してください**。
複数件の集計には `evaluate_many([(reference, hypothesis), ...])`、正規化結果の確認には `normalize(text)` を使えます。どちらも `cer_evaluator_ja` からimportできます。

## コマンドで使う

```bash
cer-evaluator-ja '(F えー)今日は晴れです(L)' '今日は晴れです'
```

結果はJSONで出力します。uv環境では先頭に `uv run` を付けて実行してください。

複数件は、次の形式で `pairs.jsonl`（UTF-8）を用意します。`id` は省略できます。

```json
{"id":"1","reference":"(F えー)今日は晴れ","hypothesis":"今日は晴れ"}
{"id":"2","reference":"(V 二人|2人)です","hypothesis":"2人です"}
```

```bash
cer-evaluator-ja --pairs pairs.jsonl > result.json
```

## 正解の注釈

| 注釈 | 扱い |
| --- | --- |
| `(D そっ)` / `(F えー)` | あってもなくてもよい |
| `(V iPhone\|アイフォーン)` / `(A です\|ですよ)` | 候補から選択 |
| `(A はい\|)` | 空文字も候補に含める |
| `(VF まあ\|まぁ)` / `(VD そっ\|そ)` | 各候補、または省略 |
| `(L)` / `(B)` / `(C)` | 無視 |

注釈は正解側のみで解釈します。[対象データセット](https://huggingface.co/datasets/bandad/asr-testset-kw-ja-v1)の `annotations.jsonl` は1行が1音声です。各行の `segments[*].text` を正解に使い、**同じ音声区間の認識結果**と比較してください。音声・区間の `excluded: true` と未確認の区間は評価対象から除きます。

CLIに渡す場合は、各区間の正解と認識結果を組にした上記の `pairs.jsonl` 形式にしてください。`annotations.jsonl` をそのまま `--pairs` に渡すことはできません。

## 主な正規化

- 英字を小文字に統一：`GPU` → `gpu`、`iPhone` → `iphone`。
- 文字幅・不要な空白・連続長音を統一。
- 数値を統一：`ゼロ` → `0`、`一万五千円` / `15,000円` → `1万5000円`。
- 電話番号のハイフン、数値範囲、cm/mmの表記を統一。
- 引用符・中黒・句読点・メール外の `!?`・小数点以外のピリオドを削除。
- 数式の記号・括弧・ハイフン・下線、小数点、メール内の記号は保持。

`イチ` → `1` の変換や読み候補の生成は行いません。変換例・例外は[正規化・評価の詳細](docs/normalization.md)を参照してください。

CERは「編集数 ÷ 採用した正解の文字数」です。複数件では、各件のCERを最小化してから編集数と文字数を合算します。正解が空の場合は分母を1として扱います。

候補数の上限は1件10万候補です。超過時はエラーになり、Pythonの `max_candidates` / CLIの `--max-candidates` で変更できます。

## 開発

```bash
uv sync
uv run pytest
uv run ruff check src tests
```
