"""Microphone capture.

Records audio in a background thread and pushes mono float32 chunks onto a
queue for a consumer (see realtime.py) to read.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Empty, Queue

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    whisper_sample_rate: int = 16000
    capture_sample_rate: int = 44100
    channels: int = 2
    chunk_seconds: float = 3.0
    # None = auto-detect a working input device.
    device: int | None = None


class MicrophoneRecorder:
    """Captures microphone audio and exposes it as mono float32 chunks."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()

        self.audio_queue: Queue[np.ndarray] = Queue()
        self.stream: sd.InputStream | None = None
        self.running = False

        self._lock = threading.Lock()

        self.selected_device: int | None = None
        self.selected_sample_rate: int | None = None
        self.selected_channels: int | None = None

    # -- device discovery -------------------------------------------------

    def _candidate_devices(self) -> list[tuple[int, int, dict]]:
        """Return (priority, index, info) for every input-capable device,
        best guesses first."""
        candidates = []
        for index, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] <= 0:
                continue

            name = info["name"].lower()
            priority = 0
            if "microphone" in name:
                priority -= 100
            if "realtek" in name:
                priority -= 50

            candidates.append((priority, index, info))

        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates

    def _find_working_device(self) -> tuple[int, int, int]:
        """Probe input devices/rates/channel counts and return the first
        combination the OS actually accepts."""
        candidates = self._candidate_devices()

        print("\n=== MICROPHONE AUTO-DETECTION ===")
        for _, index, info in candidates:
            print(
                f"Device {index}: {info['name']} | "
                f"inputs={info['max_input_channels']} | "
                f"default_rate={info['default_samplerate']}"
            )

        print("\nTesting microphone configurations...")
        for _, index, info in candidates:
            default_rate = int(info["default_samplerate"])
            rates = list(dict.fromkeys([default_rate, 44100, 48000, 16000]))

            channel_options = []
            if info["max_input_channels"] >= 2:
                channel_options.append(2)
            if info["max_input_channels"] >= 1:
                channel_options.append(1)

            for rate in rates:
                for channels in channel_options:
                    try:
                        sd.check_input_settings(
                            device=index, samplerate=rate,
                            channels=channels, dtype="float32",
                        )
                    except Exception:
                        continue

                    print(f"Candidate accepted: {index} {info['name']} "
                          f"{rate} Hz {channels} channel(s)")
                    return index, rate, channels

        raise RuntimeError(
            "No working microphone configuration was found. Check "
            "Windows microphone permissions and that a mic is connected."
        )

    # -- capture ------------------------------------------------------------

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)

        if not self.running:
            return

        audio = indata.copy() if indata.ndim == 1 else np.mean(indata, axis=1)
        self.audio_queue.put(audio.astype(np.float32))

    def start(self) -> None:
        with self._lock:
            if self.running:
                return

            self.audio_queue = Queue()

            if self.config.device is None:
                device, sample_rate, channels = self._find_working_device()
            else:
                info = sd.query_devices(self.config.device, "input")
                device = self.config.device
                sample_rate = int(info["default_samplerate"])
                channels = min(self.config.channels, int(info["max_input_channels"]))
                sd.check_input_settings(
                    device=device, samplerate=sample_rate,
                    channels=channels, dtype="float32",
                )

            self.selected_device = device
            self.selected_sample_rate = sample_rate
            self.selected_channels = channels

            print("\n=== SELECTED MICROPHONE ===")
            print("Device:", device)
            print("Name:", sd.query_devices(device, "input")["name"])
            print("Sample rate:", sample_rate)
            print("Channels:", channels)

            self.running = True
            try:
                self.stream = sd.InputStream(
                    device=device,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=int(sample_rate * self.config.chunk_seconds),
                    callback=self._callback,
                )
                self.stream.start()
                print("Microphone stream started successfully.")
            except Exception:
                self.running = False
                if self.stream is not None:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                self.stream = None
                raise

    def stop(self) -> None:
        with self._lock:
            self.running = False
            if self.stream is not None:
                try:
                    self.stream.stop()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

    def get_chunk(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self.audio_queue.get(timeout=timeout)
        except Empty:
            return None

    def drain(self) -> np.ndarray | None:
        chunks = []
        while True:
            try:
                chunks.append(self.audio_queue.get_nowait())
            except Empty:
                break
        return np.concatenate(chunks) if chunks else None

    def is_running(self) -> bool:
        return self.running
