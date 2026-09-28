"""CLIのJSON出力とエラー処理を実行して検証する。"""

import json
import subprocess
import sys


def run_cli(*args):
    """インストール済みモジュールを別プロセスで呼ぶ。"""
    return subprocess.run(
        [sys.executable, "-m", "cer_evaluator_ja", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_single_pair():
    """CLIからタグ付き正解を評価できる。"""
    completed = run_cli("(F えー)晴れ", "晴れ")
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["cer"] == 0
    assert result["normalized_reference"] == "晴れ"


def test_jsonl_with_ids_and_empty_prediction(tmp_path):
    """空の認識結果やIDを失わず、複数組を集計する。"""
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"id": "音声1", "reference": "あ", "hypothesis": ""},
        {"id": "音声2", "reference": "あいう", "hypothesis": "あいう"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    completed = run_cli("--pairs", str(path))
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["cer"] == 0.25
    assert result["results"][0]["id"] == "音声1"


def test_invalid_input_reports_line(tmp_path):
    """壊れた入力の行番号を表示し、成功扱いにしない。"""
    path = tmp_path / "bad.jsonl"
    path.write_text('\n{"reference": "晴れ"}\n', encoding="utf-8")
    completed = run_cli("--pairs", str(path))
    assert completed.returncode != 0
    assert ":2:" in completed.stderr
    assert completed.stdout == ""


def test_conflicting_arguments():
    """単一評価と一括評価の混在は拒否する。"""
    completed = run_cli("あ", "い", "--pairs", "unused.jsonl")
    assert completed.returncode != 0


def test_candidate_limit_error():
    """探索上限を超えたとき、不完全な結果を出力しない。"""
    completed = run_cli("(F あ)" * 10, "", "--max-candidates", "100")
    assert completed.returncode != 0
    assert "1,024 candidates" in completed.stderr
    assert completed.stdout == ""
