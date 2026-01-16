from typing import List, Tuple

from .types import RetrievalCandidate


def passes_confidence_gate(
    candidates: List[RetrievalCandidate],
    min_confidence: float,
    min_margin: float,
    conflict_reject: bool,
) -> Tuple[bool, float]:
    if not candidates:
        return False, 0.0

    top1 = candidates[0]
    top1_score = top1.scores.get("final", 0.0)
    if top1_score < min_confidence:
        return False, top1_score

    if len(candidates) > 1:
        top2 = candidates[1]
        top2_score = top2.scores.get("final", 0.0)
        if (top1_score - top2_score) < min_margin:
            if conflict_reject and top1.intent and top2.intent and top1.intent != top2.intent:
                return False, top1_score

    return True, top1_score
