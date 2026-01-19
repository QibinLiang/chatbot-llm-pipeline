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

## Data Preprocessing
- Convert `data/*.xlsx` conversation logs into `data/qa_pairs.jsonl` expected by `src/main.py`.
- Usage:
  - `python scripts/preprocess.py --input_dir data --output data/qa_pairs.jsonl`
  - Columns expected: `sys_response1`, `usr_query2`, `usr_intent2`, `sys_response3`, ...
  - The script emits one JSON object per line with fields: `id`, `query`, `answer`, `intent`, `context`.

## Realtime Pipeline (ASR + LLM + TTS)
- Minimal FastAPI WebSocket server that streams text frames as queries and returns JSON answers via the retrieval pipeline.
- Start server:
  - `uvicorn src.realtime.server:create_app --factory --host 0.0.0.0 --port 9000`
- WebSocket endpoint: `ws://localhost:9000/ws`
- Health check: `GET /health`
- Notes: ASR/TTS are stubbed for now (client sends text; server returns text). You can plug real ASR/TTS backends later.

### Realtime audio endpoint
- Enable ASR/TTS by installing optional deps (already in requirements): faster-whisper, edge-tts, numpy.
- Audio WebSocket: `ws://localhost:9000/ws/rt`
  - Send binary frames: PCM16LE mono, 16kHz chunks
  - Send text frame `{ "type": "flush" }` to trigger transcription + response
  - Alternatively send `{ "type": "text", "text": "你好" }` as direct query
  - Server responds with JSON answer, then streams TTS audio frames (MP3) as binary

### Non-realtime audio endpoint
- HTTP POST: `http://localhost:9000/nrt`
  - Send full audio as request body: PCM16LE mono, 16kHz
  - Server responds with TTS audio (MP3)
- Client example:
  - `python client.py --url http://127.0.0.1:9000/nrt`
