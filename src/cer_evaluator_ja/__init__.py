"""日本語の注釈付き正解に対応した正規化CER評価器。"""

from .annotations import CandidateLimitError, reference_candidates
from .evaluator import CERResult, CorpusResult, cer, evaluate, evaluate_many
from .normalization import normalize

__all__ = [
    "CERResult",
    "CandidateLimitError",
    "CorpusResult",
    "cer",
    "evaluate",
    "evaluate_many",
    "normalize",
    "reference_candidates",
]
