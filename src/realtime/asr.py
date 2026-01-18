from __future__ import annotations

import os
from typing import Optional

import numpy as np


class WhisperASR:
    """Thin wrapper around faster-whisper for bytes->text transcription.

    Expects 16kHz mono PCM16 audio bytes. Returns a single best-effort transcript.
    Lazily imports heavy deps to avoid startup cost when unused.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or os.environ.get("ASR_MODEL", "small")
        self.device = device or os.environ.get("ASR_DEVICE", "auto")
        self.compute_type = compute_type or os.environ.get("ASR_COMPUTE", "float16")
        self.language = language or os.environ.get("ASR_LANG")
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore

        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if not audio_bytes:
            return ""
        self._ensure_model()
        # Convert PCM16LE bytes -> float32 [-1,1]
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # faster-whisper expects 16kHz
        # If sample_rate != 16000, the client should resample before sending.
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
        )
        texts = [seg.text.strip() for seg in segments]
        return " ".join(t for t in texts if t)

