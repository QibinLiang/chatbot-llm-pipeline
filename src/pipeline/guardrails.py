from typing import List, Optional

from .types import AnswerPayload


def apply_guardrails(
    query: str,
    refuse_template: str,
    sensitive_keywords: List[str],
    out_of_scope_policy: str,
) -> Optional[AnswerPayload]:
    if sensitive_keywords:
        for keyword in sensitive_keywords:
            if keyword and keyword in query:
                return AnswerPayload(answer=refuse_template.strip(), citations=[], confidence=0.0, fallback=True)

    if out_of_scope_policy == "refuse" and not query.strip():
        return AnswerPayload(answer=refuse_template.strip(), citations=[], confidence=0.0, fallback=True)

    return None
