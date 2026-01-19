from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..pipeline import ChatPipeline
from ..pipeline.config import load_config
from ..pipeline.loader import load_qa_pairs
from ..pipeline.types import Message
from .asr import WhisperASR
from .tts import EdgeTTS


class SimpleASR:
    async def transcribe_stream(self, ws: WebSocket) -> AsyncIterator[str]:
        # Placeholder: expect client to send already-transcribed text frames
        while True:
            data = await ws.receive_text()
            yield data


class SimpleTTS:
    async def synthesize(self, text: str) -> bytes:
        # Placeholder: return empty audio bytes; real impl can plug a TTS backend
        return b""


def create_app(config_path: str = "config/pipeline.json", data_path: Optional[str] = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    cfg = load_config(config_path)
    src_path = data_path or cfg.get("retrieval", {}).get("index_source", "data/qa_pairs.jsonl")
    items = load_qa_pairs(src_path)
    pipeline = ChatPipeline(cfg, items)
    asr = SimpleASR()
    tts = SimpleTTS()
    whisper_asr = WhisperASR()
    edge_tts = EdgeTTS()

    @app.websocket("/ws")
    async def ws_chat(websocket: WebSocket) -> None:
        await websocket.accept()
        context: List[Message] = []
        try:
            async for text in asr.transcribe_stream(websocket):
                query = text.strip()
                if not query:
                    continue
                response = pipeline.respond(query, context)
                # send json response (text); TTS bytes could be sent on another binary frame if desired
                await websocket.send_text(json.dumps({
                    "answer": response.answer,
                    "citations": response.citations,
                    "confidence": response.confidence,
                    "fallback": response.fallback,
                }, ensure_ascii=False))
                context.append(Message(role="user", text=query))
                context.append(Message(role="system", text=response.answer))
        except WebSocketDisconnect:
            return

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "items": len(items)}

    @app.websocket("/ws/rt")
    async def ws_realtime(websocket: WebSocket) -> None:
        """Realtime endpoint:
        - Client sends binary audio frames (PCM16LE 16kHz mono) and/or JSON control frames.
        - Server transcribes, calls pipeline, and streams back TTS audio frames.
        Protocol:
          - Binary = audio chunk
          - Text JSON {"type":"flush"} to force processing and get an answer
          - Outgoing: text JSON answer; binary audio frames with synthesized TTS
        """
        await websocket.accept()
        context: List[Message] = []
        audio_buf = bytearray()
        try:
            while True:
                msg = await websocket.receive()
                if "bytes" in msg and msg["bytes"] is not None:
                    audio_buf.extend(msg["bytes"])
                    # keep collecting until flush
                    continue
                if "text" in msg and msg["text"] is not None:
                    try:
                        payload = json.loads(msg["text"]) if msg["text"] else {}
                    except Exception:
                        payload = {"type": "text", "text": msg["text"]}

                    if payload.get("type") == "flush":
                        text = whisper_asr.transcribe_bytes(bytes(audio_buf)) if audio_buf else (payload.get("text", "") or "")
                        audio_buf.clear()
                        query = (text or "").strip()
                        if not query:
                            await websocket.send_text(json.dumps({"warning": "no speech or text"}, ensure_ascii=False))
                            continue
                        response = pipeline.respond(query, context)
                        await websocket.send_text(json.dumps({
                            "answer": response.answer,
                            "citations": response.citations,
                            "confidence": response.confidence,
                            "fallback": response.fallback,
                        }, ensure_ascii=False))
                        # stream TTS back
                        async for audio in edge_tts.stream(response.answer):
                            await websocket.send_bytes(audio)
                        context.append(Message(role="user", text=query))
                        context.append(Message(role="system", text=response.answer))
                        continue
                    # If plain text query, treat as already transcribed
                    if payload.get("type") == "text":
                        query = (payload.get("text") or "").strip()
                        if not query:
                            continue
                        response = pipeline.respond(query, context)
                        await websocket.send_text(json.dumps({
                            "answer": response.answer,
                            "citations": response.citations,
                            "confidence": response.confidence,
                            "fallback": response.fallback,
                        }, ensure_ascii=False))
                        async for audio in edge_tts.stream(response.answer):
                            await websocket.send_bytes(audio)
                        context.append(Message(role="user", text=query))
                        context.append(Message(role="system", text=response.answer))
                        continue
        except WebSocketDisconnect:
            return

    @app.post("/nrt")
    async def http_non_realtime(request: Request) -> Response:
        """Non-realtime endpoint:
        - Client sends full audio (PCM16LE 16kHz mono) as HTTP body.
        - Server transcribes, calls pipeline, and returns TTS audio (MP3).
        """
        audio_bytes = await request.body()
        if not audio_bytes:
            return Response(content=b"", media_type="audio/mpeg", status_code=400)
        text = whisper_asr.transcribe_bytes(audio_bytes)
        query = (text or "").strip()
        if not query:
            return Response(content=b"", media_type="audio/mpeg", status_code=400)
        response = pipeline.respond(query, [])
        mp3_chunks: List[bytes] = []
        async for audio in edge_tts.stream(response.answer):
            mp3_chunks.append(audio)
        return Response(content=b"".join(mp3_chunks), media_type="audio/mpeg")

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=9000)
