"""実際の注釈で探索時間を測る。認識精度の評価ではない。"""

import argparse
import json
import platform
import statistics
from time import perf_counter

from cer_evaluator_ja import evaluate, normalize, reference_candidates
from cer_evaluator_ja.normalization import _normalize_prepared


def main() -> None:
    """確認済み・採用区間を評価し、候補数と処理時間をJSONで表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", help="1行1音声のannotations.jsonl（UTF-8）")
    args = parser.parse_args()
    with open(args.annotations, encoding="utf-8") as stream:
        # JSONLは1行が1音声。全体を包むannotationsキーはない。
        annotations = (json.loads(line) for line in stream if line.strip())
        texts = [
            segment["text"]
            for annotation in annotations
            if not annotation.get("excluded", False)
            for segment in annotation["segments"]
            if segment.get("confirmed", False) and not segment.get("excluded", False)
        ]
    if not texts:
        parser.error("採用区間がありません")
    counts = [sum(1 for _ in reference_candidates(text)) for text in texts]
    _normalize_prepared.cache_clear()
    start = perf_counter()
    normalize("一万円")
    warmup = perf_counter() - start
    timings = []
    start = perf_counter()
    for text in texts:
        begin = perf_counter()
        result = evaluate(text, "検証用の音声認識結果")
        if result.errors == 0:
            parser.error("固定文字列と一致する正解があり、全候補探索の条件を満たしません")
        timings.append(perf_counter() - begin)
    total = perf_counter() - start
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "segments": len(texts),
                "candidates": sum(counts),
                "max_candidates": max(counts),
                "dictionary_warmup_seconds": warmup,
                "total_seconds": total,
                "median_ms": statistics.median(timings) * 1000,
                "max_ms": max(timings) * 1000,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
