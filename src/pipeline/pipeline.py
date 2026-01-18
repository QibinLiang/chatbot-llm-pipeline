from typing import Any, Dict, List, Union

from .answerer import build_answer
from .cache import SimpleTTLCache
from .gate import passes_confidence_gate
from .guardrails import apply_guardrails
from .rerank import rerank
from .retrieval import Retriever
from .text import contains_referential, normalize_text
from .types import AnswerPayload, KnowledgeItem, Message


class ChatPipeline:
    def __init__(self, config: Dict[str, Any], items: List[KnowledgeItem]) -> None:
        self.config = config
        hybrid = config.get("retrieval", {}).get("hybrid", {})
        self.retriever = Retriever(
            items,
            bm25_top_k=hybrid.get("bm25_top_k", 20),
            vector_top_k=hybrid.get("vector_top_k", 20),
        )
        cache_cfg = config.get("cache", {})
        self.answer_cache = SimpleTTLCache(cache_cfg.get("answer_cache_ttl_sec", 900))
        self.retrieval_cache = SimpleTTLCache(cache_cfg.get("retrieval_cache_ttl_sec", 900))

    def respond(self, query: str, context: Union[List[Dict[str, str]], List[Message]]) -> AnswerPayload:
        input_cfg = self.config.get("input", {})
        ctx_cfg = input_cfg.get("context", {})
        referential_tokens = ctx_cfg.get("referential_tokens", [])
        min_len = ctx_cfg.get("min_query_len_for_context", 6)
        max_turns = ctx_cfg.get("max_turns", 4)

        normalize_cfg = input_cfg.get("normalize", {})
        normalized = normalize_text(query) if normalize_cfg.get("trim_spaces", True) else query

        guardrail = apply_guardrails(
            normalized,
            self.config.get("llm", {}).get("refuse_template", ""),
            self.config.get("guardrails", {}).get("sensitive_keywords", []),
            self.config.get("guardrails", {}).get("out_of_scope_policy", "refuse"),
        )
        if guardrail:
            return guardrail

        context_messages = self._normalize_context(context)
        combined_query = self._combine_query(normalized, context_messages, referential_tokens, min_len, max_turns)

        cached_answer = self.answer_cache.get(combined_query)
        if cached_answer:
            return cached_answer

        candidates = self.retrieval_cache.get(combined_query)
        if candidates is None:
            candidates = self.retriever.retrieve(combined_query)
            self.retrieval_cache.set(combined_query, candidates)

        if not candidates:
            return build_answer("", [], self.config.get("llm", {}).get("refuse_template", ""))

        weights = self.config.get("retrieval", {}).get("hybrid", {}).get("merge_weights", {})
        reranked = rerank(
            candidates,
            weight_vector=weights.get("vector", 0.6),
            weight_bm25=weights.get("bm25", 0.4),
            intent_boost=self.config.get("rerank", {}).get("intent_boost", 0.0),
        )

        top_k = self.config.get("rerank", {}).get("top_k", 5)
        reranked = reranked[:top_k]

        gate_cfg = self.config.get("confidence_gate", {})
        passed, confidence = passes_confidence_gate(
            reranked,
            min_confidence=gate_cfg.get("min_confidence", 0.55),
            min_margin=gate_cfg.get("min_margin", 0.05),
            conflict_reject=gate_cfg.get("conflict_reject", True),
        )
        if not passed:
            return build_answer("", [], self.config.get("llm", {}).get("refuse_template", ""))

        response = build_answer(normalized, reranked, self.config.get("llm", {}).get("refuse_template", ""))
        response.confidence = confidence
        self.answer_cache.set(combined_query, response)
        return response

    @staticmethod
    def _normalize_context(context: Union[List[Dict[str, str]], List[Message]]) -> List[Message]:
        messages: List[Message] = []
        for msg in context:
            if isinstance(msg, Message):
                messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role")
                text = msg.get("text")
                if role and text:
                    messages.append(Message(role=role, text=text))
        return messages

    @staticmethod
    def _combine_query(
        query: str,
        context: List[Message],
        referential_tokens: List[str],
        min_len: int,
        max_turns: int,
    ) -> str:
        use_context = len(query) < min_len or contains_referential(query, referential_tokens)
        if not context or not use_context:
            return query

        user_texts = [msg.text for msg in context if msg.role == "user"]
        tail = user_texts[-max_turns:] if max_turns > 0 else user_texts
        if not tail:
            return query
        return " ".join(tail + [query])
