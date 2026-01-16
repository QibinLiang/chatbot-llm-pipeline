import json
from pathlib import Path
from typing import List

from .types import KnowledgeItem, Message


def load_qa_pairs(path: str) -> List[KnowledgeItem]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"QA data not found: {data_path}")

    items: List[KnowledgeItem] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            context = [Message(**msg) for msg in record.get("context", [])]
            items.append(
                KnowledgeItem(
                    id=record.get("id", ""),
                    query=record.get("query", ""),
                    answer=record.get("answer", ""),
                    intent=record.get("intent"),
                    context=context,
                )
            )
    return items
