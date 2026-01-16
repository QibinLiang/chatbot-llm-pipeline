from typing import List, Tuple

from .types import RetrievalCandidate


def rerank(
    candidates: List[RetrievalCandidate],
    weight_vector: float,
    weight_bm25: float,
    intent_boost: float,
) -> List[RetrievalCandidate]:
    if not candidates:
        return []

    max_bm25 = max(c.scores.get("bm25", 0.0) for c in candidates) or 1.0
    for cand in candidates:
        bm25_norm = cand.scores.get("bm25", 0.0) / max_bm25
        vector_score = cand.scores.get("vector", 0.0)
        intent_score = intent_boost if cand.intent else 0.0
        cand.scores["final"] = weight_vector * vector_score + weight_bm25 * bm25_norm + intent_score

    return sorted(candidates, key=lambda c: c.scores.get("final", 0.0), reverse=True)
