# Chatbot LLM Pipeline

Minimal scaffold for an LLM-powered chatbot pipeline.

## Goals
- Provide a clean starting point for prompt orchestration, retrieval, and evaluation.
- Keep configuration and secrets outside of source control.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Project Structure
- `src/` application code
- `tests/` test suites
- `data/` local datasets (ignored by git)
- `docs/` design notes

## Notes
- Put secrets in `.env` (not committed).
- Update `requirements.txt` as dependencies are added.
