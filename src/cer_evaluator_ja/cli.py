"""1組のテキスト、またはUTF-8 JSONL形式の複数組を評価するCLI。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .evaluator import evaluate, evaluate_many


def main() -> None:
    """引数を検証し、評価結果をJSONとして標準出力に書き出す。"""
    parser = argparse.ArgumentParser(description="注釈付き正解テキストの最小正規化CER")
    parser.add_argument("reference", nargs="?", help="正解テキスト（注釈付き）")
    parser.add_argument("hypothesis", nargs="?", help="音声認識結果（プレーンテキスト）")
    parser.add_argument("--pairs", type=Path, help="reference/hypothesisを含むUTF-8 JSONL")
    parser.add_argument("--max-candidates", type=int, default=100_000)
    args = parser.parse_args()
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if args.pairs is not None:
        if args.reference is not None or args.hypothesis is not None:
            parser.error("--pairs cannot be combined with positional texts")
    elif args.reference is None or args.hypothesis is None:
        parser.error("provide reference and hypothesis, or --pairs")
    try:
        if args.pairs is None:
            result = asdict(
                evaluate(args.reference, args.hypothesis, max_candidates=args.max_candidates)
            )
        else:
            pairs = []
            ids = []
            with args.pairs.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("expected a JSON object")
                        if not all(
                            isinstance(row.get(k), str) for k in ("reference", "hypothesis")
                        ):
                            raise ValueError("reference and hypothesis must be strings")
                        pairs.append((row["reference"], row["hypothesis"]))
                        ids.append(row.get("id"))
                    except ValueError as exc:
                        raise ValueError(f"{args.pairs}:{line_number}: {exc}") from exc
            result = asdict(evaluate_many(pairs, max_candidates=args.max_candidates))
            for row, identifier in zip(result["results"], ids):
                if identifier is not None:
                    row["id"] = identifier
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
