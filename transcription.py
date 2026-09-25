"""Speech-to-text via faster-whisper."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from faster_whisper import WhisperModel

# Chunks quieter than this (peak amplitude, float32 range [-1, 1]) are
# treated as silence and skipped, to avoid sending noise to Whisper.
SILENCE_PEAK_THRESHOLD = 0.003


@lru_cache(maxsize=6)
def get_whisper_model(model_size: str = "base", device: str = "cpu") -> WhisperModel:
    """Load (and cache) a faster-whisper model. Compute type is derived
    from the device: float16 on cuda, int8 on cpu."""
    if device == "cuda":
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    print(f"Loading Whisper model: {model_size} | device={device} | compute={compute_type}")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio_array(
    audio: np.ndarray,
    model_size: str = "base",
    device: str = "cpu",
    language: str | None = None,
) -> str:
    """Transcribe a mono float32 numpy array (already at the model's
    expected sample rate) and return the text, or "" for silence/empty."""
    if audio is None or len(audio) == 0:
        return ""

    audio = np.asarray(audio, dtype=np.float32)
    if float(np.max(np.abs(audio))) < SILENCE_PEAK_THRESHOLD:
        return ""

    model = get_whisper_model(model_size, device)
    segments, _info = model.transcribe(
        audio,
        language=None if language in (None, "", "Auto") else language,
        beam_size=1,
        best_of=1,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()


def transcribe_file(
    path: str,
    model_size: str = "base",
    device: str = "cpu",
    language: str | None = None,
) -> str:
    """Transcribe an audio file on disk and return the text."""
    model = get_whisper_model(model_size, device)
    segments, _info = model.transcribe(
        path,
        language=None if language in (None, "", "Auto") else language,
        beam_size=1,
        vad_filter=True,
    )
    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
