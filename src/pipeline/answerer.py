from typing import List

from .types import AnswerPayload, RetrievalCandidate


def build_answer(
    question: str,
    candidates: List[RetrievalCandidate],
    refuse_template: str,
) -> AnswerPayload:
    if not candidates:
        return AnswerPayload(answer=refuse_template.strip(), citations=[], confidence=0.0, fallback=True)

    top = candidates[0]
    formatted = f"答复：{top.answer}\n依据：{top.answer}\n生效时间："
    return AnswerPayload(
        answer=formatted,
        citations=[top.id],
        confidence=top.scores.get("final", 0.0),
        fallback=False,
    )
