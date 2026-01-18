"""
Preprocess XLSX conversation data into JSONL knowledge items.

Input assumptions (based on data/conversation_data.xlsx):
- Columns include alternating system/user turns like:
  sys_response1, usr_query2, usr_intent2, sys_response3, usr_query4, ... , sys_response17
- Each row represents one dialogue; generate a QA pair for each user query that has a following system response.

Output JSONL schema (one object per line):
{
  "id": str,
  "query": str,
  "answer": str,
  "intent": str | null,
  "context": [{"role": "system"|"user", "text": str}, ...]
}

Usage:
  python scripts/preprocess.py --input_dir data --output data/qa_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _to_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x)
    return s.strip()


def _build_items_from_row(row: pd.Series, row_idx: int, source_name: str) -> List[Dict]:
    items: List[Dict] = []
    row_id_base = _to_str(row.get("序号")) or f"{source_name}-{row_idx}"

    # Collect conversation context incrementally
    context: List[Dict[str, str]] = []

    # Seed with first system response if present
    sys1 = _to_str(row.get("sys_response1"))
    if sys1:
        context.append({"role": "system", "text": sys1})

    # We expect even user turns (2,4,...) and following odd system responses (3,5,...)
    # Detect all columns and determine max step
    steps: List[int] = []
    for col in row.index:
        if isinstance(col, str) and col.startswith("usr_query"):
            try:
                step = int(col.replace("usr_query", ""))
                steps.append(step)
            except ValueError:
                pass
    steps = sorted(set(steps))

    for step in steps:
        # Current user turn
        q_col = f"usr_query{step}"
        i_col = f"usr_intent{step}"
        a_col = f"sys_response{step + 1}"

        query = _to_str(row.get(q_col))
        intent = _to_str(row.get(i_col)) or None
        answer = _to_str(row.get(a_col))

        # Skip if no valid pair
        if not query or not answer:
            # If there is a user query but no answer, still update context with the query
            # so later turns include it when applicable.
            if query:
                context.append({"role": "user", "text": query})
            continue

        # Build item with a snapshot of context BEFORE this query
        item_context = list(context) if context else []
        item_id = f"{row_id_base}#t{step}"
        item = {
            "id": item_id,
            "query": query,
            "answer": answer,
            "intent": intent,
            "context": item_context,
        }
        items.append(item)

        # Update rolling context with this turn
        context.append({"role": "user", "text": query})
        context.append({"role": "system", "text": answer})

    return items


def process_file(path: Path) -> List[Dict]:
    df = pd.read_excel(path)
    items: List[Dict] = []
    for idx, row in df.iterrows():
        items.extend(_build_items_from_row(row, idx, path.stem))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XLSX conversation data to JSONL QA pairs.")
    parser.add_argument("--input_dir", default="data", help="Directory containing .xlsx files")
    parser.add_argument("--output", default="data/qa_pairs.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted([p for p in input_dir.glob("*.xlsx") if p.is_file()])
    if not xlsx_files:
        print(f"No .xlsx files found in {input_dir}")
        return

    all_items: List[Dict] = []
    for f in xlsx_files:
        all_items.extend(process_file(f))

    with output_path.open("w", encoding="utf-8") as w:
        for obj in all_items:
            json.dump(obj, w, ensure_ascii=False)
            w.write("\n")

    print(f"Wrote {len(all_items)} items to {output_path}")


if __name__ == "__main__":
    main()

