#!/usr/bin/env python3
"""
Interactive WebSocket client for realtime testing (录音->按's'保存->等待->播放)。

支持：
- 文本模式（/ws）：发送文本，打印 JSON。
- 实时语音模式（/ws/rt）：麦克风采集 16k 单声道 PCM16 流，按 's' 发送 flush，等待答复并播放返回的 TTS 音频。
- 非实时语音模式（HTTP /nrt）：按 's' 开始录音，再按 's' 停止并发送整段语音，收到后播放返回的 TTS 音频。

示例：
  python client.py --url ws://127.0.0.1:9000/ws --query "开发票"
  python client.py --url ws://127.0.0.1:9000/ws               # 文本交互
  python client.py --url ws://127.0.0.1:9000/ws/rt            # 语音交互（按 's' 发送）
  python client.py --url http://127.0.0.1:9000/nrt            # 非实时语音交互（按 's' 开始/停止）
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import termios
import tty
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import websockets
try:
    import sounddevice as sd  # type: ignore
except Exception as _e:
    sd = None  # type: ignore


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


def _play_mp3_bytes(mp3_bytes: bytes) -> bool:
    """Try to play MP3 bytes. Prefer pydub+simpleaudio, else ffplay, else save file only.
    Returns True if played, False otherwise.
    """
    # Try pydub + simpleaudio
    try:
        from pydub import AudioSegment  # type: ignore
        import simpleaudio as sa  # type: ignore
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
        play_obj.wait_done()
        return True
    except Exception:
        pass
    # Try ffplay
    if shutil.which("ffplay"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(mp3_bytes)
            tmp = f.name
        try:
            subprocess.run(["ffplay", "-autoexit", "-nodisp", tmp], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
    return False


class KeyWatcher:
    def __init__(self, trigger_key: str = "s") -> None:
        self.trigger_key = trigger_key
        self._thread: Optional[threading.Thread] = None

    def start(self, event: threading.Event) -> None:
        self._thread = threading.Thread(target=self._run, args=(event,), daemon=True)
        self._thread.start()

    def _run(self, event: threading.Event) -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                if ch.lower() == self.trigger_key:
                    event.set()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def audio_client_realtime(uri: str, headers: List[tuple[str, str]], chunk_ms: int = 20) -> None:
    if sd is None:
        print("sounddevice 未安装，无法进行语音交互。请先 pip install sounddevice", file=sys.stderr)
        return
    samplerate = 16000
    channels = 1
    blocksize = max(1, int(samplerate * chunk_ms / 1000))  # samples per block
    q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)

    def _callback(indata, frames, time, status):  # type: ignore
        if status:
            # print status to stderr but don't spam
            pass
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            try:
                q.get_nowait()
            except Exception:
                pass
            q.put_nowait(bytes(indata))

    print("连接中... 按 's' 发送(保存)并等待结果，Ctrl+C 退出。")
    async with websockets.connect(uri, extra_headers=headers, max_size=None) as ws:
        while True:
            flush_event = threading.Event()
            watcher = KeyWatcher('s')
            watcher.start(flush_event)
            print("开始录音... (按 's' 发送)")
            with sd.RawInputStream(samplerate=samplerate, channels=channels, dtype='int16', blocksize=blocksize, callback=_callback):
                # stream audio until 's'
                try:
                    while not flush_event.is_set():
                        try:
                            chunk = q.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        await ws.send(chunk)
                except KeyboardInterrupt:
                    print("收到中断，退出。")
                    return
            # send flush
            await ws.send(json.dumps({"type": "flush"}))
            print("已发送，等待应答...")

            # First expect JSON answer
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                print("等待应答超时。")
                continue
            mp3_chunks: List[bytes] = []
            if isinstance(msg, str):
                print(msg)
            else:
                # unexpected binary first
                mp3_chunks.append(msg)
                # then expect text
                while True:
                    m = await ws.recv()
                    if isinstance(m, str):
                        print(m)
                        break
                    mp3_chunks.append(m)

            # collect audio with short timeout
            while True:
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=0.6)
                except asyncio.TimeoutError:
                    break
                if isinstance(m, bytes):
                    mp3_chunks.append(m)
                else:
                    # received text - treat as new event; print and stop collecting audio
                    print(m)
                    break

            if mp3_chunks:
                mp3_data = b"".join(mp3_chunks)
                played = _play_mp3_bytes(mp3_data)
                if not played:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                        f.write(mp3_data)
                        tmp = f.name
                    print(f"已保存 TTS 音频到: {tmp}")
            # next turn or quit
            print("按 Enter 开始下一轮，输入 q 回车退出：", end="", flush=True)
            try:
                line = sys.stdin.readline().strip()
            except KeyboardInterrupt:
                line = 'q'
            if line.lower() == 'q':
                break


def _post_audio_http(url: str, audio_bytes: bytes, headers: List[tuple[str, str]]) -> bytes:
    req = urllib.request.Request(url, data=audio_bytes, method="POST")
    req.add_header("Content-Type", "audio/pcm")
    for key, value in headers:
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        print(f"HTTP {e.code}: {body[:200]!r}", file=sys.stderr)
        return b""
    except Exception as e:
        print(f"HTTP 请求失败: {e}", file=sys.stderr)
        return b""


async def audio_client_non_realtime_http(uri: str, headers: List[tuple[str, str]]) -> None:
    if sd is None:
        print("sounddevice 未安装，无法进行语音交互。请先 pip install sounddevice", file=sys.stderr)
        return
    samplerate = 16000
    channels = 1
    blocksize = 1024  # samples per block
    q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

    def _callback(indata, frames, time, status):  # type: ignore
        if status:
            pass
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            try:
                q.get_nowait()
            except Exception:
                pass
            q.put_nowait(bytes(indata))

    print("按 's' 开始录音，再按 's' 停止并发送，Ctrl+C 退出。")
    while True:
        start_event = threading.Event()
        KeyWatcher('s').start(start_event)
        print("等待开始...")
        while not start_event.is_set():
            await asyncio.sleep(0.05)

        stop_event = threading.Event()
        KeyWatcher('s').start(stop_event)
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                break
        audio_buf = bytearray()
        print("开始录音... (按 's' 停止)")
        with sd.RawInputStream(samplerate=samplerate, channels=channels, dtype='int16', blocksize=blocksize, callback=_callback):
            try:
                while not stop_event.is_set():
                    try:
                        chunk = q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    audio_buf.extend(chunk)
            except KeyboardInterrupt:
                print("收到中断，退出。")
                return

        if not audio_buf:
            print("未录到音频，重试。")
            continue

        print("录音完成，发送中...")
        mp3_data = await asyncio.to_thread(_post_audio_http, uri, bytes(audio_buf), headers)
        if mp3_data:
            played = _play_mp3_bytes(mp3_data)
            if not played:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    f.write(mp3_data)
                    tmp = f.name
                print(f"已保存 TTS 音频到: {tmp}")
        else:
            print("未收到音频响应。")
        print("按 Enter 开始下一轮，输入 q 回车退出：", end="", flush=True)
        try:
            line = sys.stdin.readline().strip()
        except KeyboardInterrupt:
            line = 'q'
        if line.lower() == 'q':
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for realtime chatbot pipeline")
    parser.add_argument("--url", required=True, help="URL, e.g. ws://host:9000/ws, ws://host:9000/ws/rt, or http://host:9000/nrt")
    parser.add_argument("--query", default=None, help="Text mode: one-shot query. Omit to enter interactive mode.")
    parser.add_argument("--chunk-ms", type=int, default=20, help="Audio chunk size (ms) in realtime mode")
    parser.add_argument("--auth", default=None, help="Authorization header if needed, e.g. 'Bearer xxx'")
    args = parser.parse_args()

    headers = _build_headers(args)
    if args.url.endswith("/ws"):
        asyncio.run(text_client(args.url, args.query, headers))
        return
    if args.url.endswith("/ws/rt"):
        asyncio.run(audio_client_realtime(args.url, headers, chunk_ms=args.chunk_ms))
        return
    if args.url.startswith("http") and args.url.endswith("/nrt"):
        asyncio.run(audio_client_non_realtime_http(args.url, headers))
        return
    print("Unknown endpoint. Use /ws for text, /ws/rt for realtime audio, or http(s)://.../nrt for non-realtime audio.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
