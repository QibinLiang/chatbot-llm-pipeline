import re
from typing import Iterable, List

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = text.replace("\u3000", " ").strip().lower()
    normalized = SPACE_RE.sub(" ", normalized)
    return normalized


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for part in TOKEN_RE.findall(text):
        if CJK_RE.fullmatch(part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part.lower())
    return tokens


def contains_referential(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)
