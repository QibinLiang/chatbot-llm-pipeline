from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional


class EdgeTTS:
    """Thin wrapper over edge-tts to stream TTS audio.

    Produces MP3 bytes; clients can play progressively.
    """

    def __init__(self, voice: Optional[str] = None) -> None:
        self.voice = voice or "zh-CN-XiaoxiaoNeural"

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        if not text:
            return
        try:
            import edge_tts  # type: ignore
        except Exception:
            # Fallback: emit nothing if edge-tts not installed
            return
        communicate = edge_tts.Communicate(text, self.voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

