"""Ties microphone capture, transcription, and NLP extraction together
into a live, continuously-updating meeting state."""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.signal import resample_poly

from .action_items import extract_action_items
from .audio import AudioConfig, MicrophoneRecorder
from .decisions import extract_decisions
from .inference import summarize
from .topics import extract_topics
from .transcription import transcribe_audio_array

# Chunks quieter than this (peak amplitude) are skipped before ever
# reaching Whisper, to save compute on dead air.
SILENCE_PEAK_THRESHOLD = 0.00001


class LiveMeetingState:
    """Shared, lock-protected state read by the UI and written by the
    background worker thread."""

    def __init__(self):
        self.active = False
        self.transcript = ""
        self.live_summary = ""
        self.topics: list[str] = []
        self.actions: list[dict] = []
        self.decisions: list[str] = []
        self.transcript_chunks: list[str] = []
        self.last_peak: float = 0.0  # most recent chunk's peak amplitude, for diagnostics
        self.error: str | None = None

        self._lock = threading.Lock()


class LiveMeetingEngine:
    """Runs microphone capture + transcription + NLP extraction on a
    background thread and exposes thread-safe snapshots of the result."""

    # Tracks the most recently started engine across the whole process, so
    # that a stale engine left running from a previous Streamlit session
    # (e.g. after a browser refresh, which resets session_state but not
    # background threads) is always stopped before a new one claims the
    # microphone. Without this, two engines can end up fighting over the
    # same audio device and neither gets clean audio.
    _active_instance: "LiveMeetingEngine | None" = None

    def __init__(
        self,
        model_path: str,
        whisper_model: str = "base",
        whisper_device: str = "cpu",
        compute: str = "int8",  # kept for backward-compat / display purposes
        chunk_seconds: float = 5,
        summary_interval: float = 30,
        language: str = "Auto",
        microphone_device: int | None = None,
    ):
        self.model_path = model_path
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.compute = compute
        self.summary_interval = summary_interval
        self.language = language

        self.audio_config = AudioConfig(
            capture_sample_rate=44100,
            whisper_sample_rate=16000,
            channels=2,
            chunk_seconds=chunk_seconds,
            device=microphone_device,  # None = auto-detect a working mic
        )
        self.recorder = MicrophoneRecorder(self.audio_config)

        self.state = LiveMeetingState()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.last_summary_time = 0.0

    # -- audio helpers --------------------------------------------------

    def _resample_to_whisper_rate(self, audio: np.ndarray | None) -> np.ndarray | None:
        if audio is None or len(audio) == 0:
            return None

        input_rate = self.audio_config.capture_sample_rate
        output_rate = self.audio_config.whisper_sample_rate
        audio = audio.astype(np.float32)

        if input_rate == output_rate:
            return audio

        gcd = np.gcd(input_rate, output_rate)
        up, down = output_rate // gcd, input_rate // gcd
        return resample_poly(audio, up, down).astype(np.float32)

    # -- worker loop ------------------------------------------------------

    def _process_chunk(self, chunk: np.ndarray) -> None:
        audio = self._resample_to_whisper_rate(chunk)
        if audio is None:
            return

        peak = float(np.max(np.abs(audio)))
        with self.state._lock:
            self.state.last_peak = peak

        if peak < SILENCE_PEAK_THRESHOLD:
            return

        text = transcribe_audio_array(
            audio,
            model_size=self.whisper_model,
            device=self.whisper_device,
            language=self.language,
        )
        text = (text or "").strip()
        if not text:
            return

        with self.state._lock:
            self.state.transcript_chunks.append(text)
            self.state.transcript = (
                f"{self.state.transcript} {text}".strip() if self.state.transcript else text
            )
            transcript = self.state.transcript

            # Same extraction logic used in the final report, so the live
            # view and the end-of-meeting report never disagree.
            self.state.topics = extract_topics(transcript)
            self.state.actions = extract_action_items(transcript)
            self.state.decisions = extract_decisions(transcript)

        self._maybe_refresh_live_summary(transcript)

    def _maybe_refresh_live_summary(self, transcript: str) -> None:
        now = time.time()
        if now - self.last_summary_time < self.summary_interval:
            return

        try:
            summary = summarize(transcript, model_path=self.model_path,
                                 max_length=130, min_length=30)
            with self.state._lock:
                self.state.live_summary = summary
        except Exception as error:
            with self.state._lock:
                self.state.error = f"Live summary error: {error}"

        self.last_summary_time = now

    def _worker(self) -> None:
        self.last_summary_time = time.time()

        while not self.stop_event.is_set():
            try:
                chunk = self.recorder.get_chunk(timeout=0.5)
                if chunk is not None:
                    self._process_chunk(chunk)
            except Exception as error:
                with self.state._lock:
                    self.state.error = f"Live transcription error: {error}"
                time.sleep(0.5)

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        if self.state.active:
            return

        # Guard against a stale engine (e.g. left running from a previous,
        # since-refreshed Streamlit session) still holding the microphone.
        stale = LiveMeetingEngine._active_instance
        if stale is not None and stale is not self and stale.state.active:
            print("Stopping a stale, still-running meeting engine before starting a new one.")
            stale.stop()

        with self.state._lock:
            self.state.error = None
            self.state.active = True

        self.stop_event.clear()

        try:
            self.recorder.start()
        except Exception:
            with self.state._lock:
                self.state.active = False
            raise

        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        LiveMeetingEngine._active_instance = self

    def stop(self) -> None:
        if not self.state.active:
            return

        self.stop_event.set()
        self.recorder.stop()

        if self.worker_thread is not None:
            self.worker_thread.join(timeout=3)

        with self.state._lock:
            self.state.active = False

        if LiveMeetingEngine._active_instance is self:
            LiveMeetingEngine._active_instance = None

    def get_transcript(self) -> str:
        with self.state._lock:
            return self.state.transcript

    def snapshot(self) -> dict:
        with self.state._lock:
            return {
                "active": self.state.active,
                "transcript": self.state.transcript,
                "live_summary": self.state.live_summary,
                "topics": list(self.state.topics),
                "actions": list(self.state.actions),
                "decisions": list(self.state.decisions),
                "transcript_chunks": list(self.state.transcript_chunks),
                "last_peak": self.state.last_peak,
                "error": self.state.error,
            }
