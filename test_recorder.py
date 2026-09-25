"""Exercises MicrophoneRecorder end-to-end: starts capture, collects 10
seconds of chunks, and saves the result to a WAV file for playback."""

import time

import numpy as np
import soundfile as sf

from src.audio import AudioConfig, MicrophoneRecorder

DURATION_SECONDS = 10
OUTPUT_PATH = "recorder_test.wav"

config = AudioConfig(capture_sample_rate=44100, whisper_sample_rate=16000,
                      channels=2, chunk_seconds=5, device=2)
recorder = MicrophoneRecorder(config)

print("Starting microphone...")
recorder.start()
print(f"Speak now for {DURATION_SECONDS} seconds...")

chunks = []
start = time.time()
while time.time() - start < DURATION_SECONDS:
    chunk = recorder.get_chunk(timeout=1)
    if chunk is not None:
        print("Received chunk:", len(chunk), "samples", "peak:", float(np.max(np.abs(chunk))))
        chunks.append(chunk)

recorder.stop()
print("Recording stopped.")

if chunks:
    audio = np.concatenate(chunks)
    print("Total samples:", len(audio))
    print("Maximum amplitude:", float(np.max(np.abs(audio))))

    # Use the recorder's actual capture rate, not a hardcoded guess, so the
    # saved WAV plays back at the correct speed/pitch.
    sf.write(OUTPUT_PATH, audio, recorder.selected_sample_rate or config.capture_sample_rate)
    print("Saved", OUTPUT_PATH)
else:
    print("ERROR: No audio chunks received.")
