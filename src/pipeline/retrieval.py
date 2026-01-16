import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .text import normalize_text, tokenize
from .types import KnowledgeItem, RetrievalCandidate


class BM25Index:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avgdl = 0.0
        self.term_freqs: List[Counter] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.doc_tokens: List[List[str]] = []

        for idx, doc in enumerate(documents):
            tokens = tokenize(normalize_text(doc))
            self.doc_tokens.append(tokens)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_len.append(len(tokens))
            for term, freq in tf.items():
                self.postings[term].append((idx, freq))

        total_docs = len(documents)
        self.avgdl = sum(self.doc_len) / total_docs if total_docs else 0.0
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str) -> List[float]:
        scores = [0.0] * len(self.doc_tokens)
        tokens = tokenize(normalize_text(query))
        for term in tokens:
            if term not in self.postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, freq in self.postings[term]:
                dl = self.doc_len[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_idx] += idf * (freq * (self.k1 + 1)) / denom
        return scores

    def token_sets(self) -> List[set]:
        return [set(tokens) for tokens in self.doc_tokens]


class Retriever:
    def __init__(self, items: List[KnowledgeItem], bm25_top_k: int, vector_top_k: int) -> None:
        self.items = items
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self._texts = [build_retrieval_text(item) for item in items]
        self._bm25 = BM25Index(self._texts)
        self._token_sets = self._bm25.token_sets()

    def retrieve(self, query: str) -> List[RetrievalCandidate]:
        bm25_scores = self._bm25.score(query)
        top_bm25 = sorted(range(len(bm25_scores)), key=bm25_scores.__getitem__, reverse=True)[: self.bm25_top_k]

        query_tokens = set(tokenize(normalize_text(query)))
        vector_scores = []
        for idx, token_set in enumerate(self._token_sets):
            if not token_set or not query_tokens:
                score = 0.0
            else:
                score = len(query_tokens & token_set) / len(query_tokens | token_set)
            vector_scores.append(score)

        top_vector = sorted(range(len(vector_scores)), key=vector_scores.__getitem__, reverse=True)[: self.vector_top_k]

        merged = set(top_bm25) | set(top_vector)
        candidates: List[RetrievalCandidate] = []
        for idx in merged:
            item = self.items[idx]
            candidates.append(
                RetrievalCandidate(
                    id=item.id,
                    answer=item.answer,
                    intent=item.intent,
                    scores={
                        "bm25": bm25_scores[idx],
                        "vector": vector_scores[idx],
                    },
                )
            )
        return candidates


def build_retrieval_text(item: KnowledgeItem) -> str:
    parts = [msg.text for msg in item.context if msg.role == "user"]
    parts.append(item.query)
    return " ".join(parts)
