"""Entry point for the chatbot pipeline demo."""

import argparse
from pathlib import Path

from pipeline import ChatPipeline
from pipeline.config import load_config
from pipeline.loader import load_qa_pairs
from pipeline.types import Message


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local chatbot pipeline demo.")
    parser.add_argument("--config", default="config/pipeline.json", help="Path to config file.")
    parser.add_argument("--data", default=None, help="Path to QA jsonl file.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = args.data or config.get("retrieval", {}).get("index_source", "")
    if not data_path:
        print("No data source configured. Set retrieval.index_source or --data.")
        return
    if not Path(data_path).exists():
        print(f"QA data not found: {data_path}")
        print("Generate data/qa_pairs.jsonl first or pass --data.")
        return

    items = load_qa_pairs(data_path)
    pipeline = ChatPipeline(config, items)

    context: list = []
    print("Chatbot pipeline ready. Type 'exit' to quit.")
    while True:
        user_input = input("you> ").strip()
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        response = pipeline.respond(user_input, context)
        print(f"bot> {response.answer}")
        context.append(Message(role="user", text=user_input))
        context.append(Message(role="system", text=response.answer))


if __name__ == "__main__":
    main()
