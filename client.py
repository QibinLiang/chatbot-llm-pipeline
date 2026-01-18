#!/usr/bin/env python3
"""
Simple WebSocket client for testing the realtime pipeline server.

Supports:
- Text mode (`/ws`): send text queries and print JSON responses.
- Audio mode (`/ws/rt`): stream WAV audio, flush, receive JSON answer and TTS audio (optionally saved).

Examples:
  python client.py --url ws://127.0.0.1:9000/ws --query "开发票"
  python client.py --url ws://127.0.0.1:9000/ws               # interactive
  python client.py --url ws://127.0.0.1:9000/ws/rt --audio sample.wav --save-audio out.mp3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
import audioop
from typing import Dict, List, Optional

import websockets


def _build_headers(args: argparse.Namespace) -> List[tuple[str, str]]:
    headers: List[tuple[str, str]] = []
    if args.auth:
        headers.append(("Authorization", args.auth))
    return headers


async def text_client(uri: str, query: Optional[str], headers: List[tuple[str, str]]):
    async with websockets.connect(uri, extra_headers=headers) as ws:
        if query is not None:
            await ws.send(query)
            resp = await ws.recv()
            print(resp)
            return
        print("Connected. Type 'exit' to quit.")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in {"exit", "quit"}:
                break
            await ws.send(text)
            resp = await ws.recv()
            try:
                obj = json.loads(resp)
            except Exception:
                obj = {"raw": resp}
            print("bot>", json.dumps(obj, ensure_ascii=False))


def _load_wav_to_pcm16_mono_16k(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        nch, sw, sr, nframes, _, _ = wf.getparams()
        data = wf.readframes(nframes)
    # convert to mono
    if nch != 1:
        data = audioop.tomono(data, sw, 0.5, 0.5)
        nch = 1
    # convert sample width to 2 bytes
    if sw != 2:
        data = audioop.lin2lin(data, sw, 2)
        sw = 2
    # resample to 16k
    if sr != 16000:
        data, _ = audioop.ratecv(data, 2, 1, sr, 16000, None)
        sr = 16000
    return data


async def audio_client(
    uri: str,
    audio_path: Optional[str],
    save_audio: Optional[str],
    chunk_ms: int,
    headers: List[tuple[str, str]],
    text_query: Optional[str] = None,
):
    async with websockets.connect(uri, extra_headers=headers, max_size=None) as ws:
        if text_query:
            await ws.send(json.dumps({"type": "text", "text": text_query}, ensure_ascii=False))
        elif audio_path:
            pcm = _load_wav_to_pcm16_mono_16k(audio_path)
            bytes_per_ms = 16000 * 2 // 1000  # 32 bytes per ms
            sz = max(1, bytes_per_ms * max(1, int(chunk_ms)))
            for i in range(0, len(pcm), sz):
                await ws.send(pcm[i : i + sz])
            await ws.send(json.dumps({"type": "flush"}))
        else:
            print("Either --audio or --query is required in audio mode.")
            return

        # Expect JSON answer first
        msg = await ws.recv()
        if isinstance(msg, bytes):
            # Unexpected binary first; collect and wait for text
            audio_chunks = [msg]
            while True:
                m = await ws.recv()
                if isinstance(m, str):
                    print(m)
                    break
                audio_chunks.append(m)
            if save_audio:
                with open(save_audio, "wb") as f:
                    for ch in audio_chunks:
                        f.write(ch)
                print(f"Saved audio -> {save_audio}")
            return
        else:
            print(msg)
        # Then collect audio with timeout
        audio_collected: List[bytes] = []
        if save_audio:
            while True:
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    break
                if isinstance(m, bytes):
                    audio_collected.append(m)
                else:
                    # another text message; stop collecting
                    break
            if audio_collected:
                with open(save_audio, "wb") as f:
                    for ch in audio_collected:
                        f.write(ch)
                print(f"Saved audio -> {save_audio}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for realtime chatbot pipeline")
    parser.add_argument("--url", required=True, help="WebSocket URL, e.g. ws://host:9000/ws or /ws/rt")
    parser.add_argument("--query", default=None, help="One-shot query text. If omitted in /ws, enter interactive mode.")
    parser.add_argument("--audio", default=None, help="Path to WAV audio (for /ws/rt)")
    parser.add_argument("--save-audio", default=None, help="Where to save returned TTS (MP3)")
    parser.add_argument("--chunk-ms", type=int, default=240, help="Chunk size in ms when streaming audio")
    parser.add_argument("--auth", default=None, help="Authorization header if needed, e.g. 'Bearer xxx'")
    args = parser.parse_args()

    headers = _build_headers(args)
    if args.url.endswith("/ws"):
        asyncio.run(text_client(args.url, args.query, headers))
        return
    if args.url.endswith("/ws/rt"):
        asyncio.run(audio_client(args.url, args.audio, args.save_audio, args.chunk_ms, headers, text_query=args.query))
        return
    print("Unknown endpoint. Use /ws for text or /ws/rt for audio.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()

