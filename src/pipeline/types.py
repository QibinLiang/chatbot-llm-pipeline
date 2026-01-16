from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Message:
    role: str
    text: str


@dataclass
class KnowledgeItem:
    id: str
    query: str
    answer: str
    intent: Optional[str] = None
    context: List[Message] = field(default_factory=list)


@dataclass
class RetrievalCandidate:
    id: str
    answer: str
    intent: Optional[str]
    scores: Dict[str, float]


@dataclass
class AnswerPayload:
    answer: str
    citations: List[str]
    confidence: float
    fallback: bool = False
