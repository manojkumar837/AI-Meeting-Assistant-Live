"""Records a fixed-length clip from a specific device and saves it to a WAV
file for playback verification."""

import sounddevice as sd
import soundfile as sf

DEVICE = 9
SAMPLE_RATE = 48000
DURATION = 10
OUTPUT_PATH = "test_microphone.wav"

print("Selected microphone:")
print(sd.query_devices(DEVICE))

print(f"\nRecording for {DURATION} seconds...")
print(">>> SPEAK NOW <<<")

audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
               channels=1, dtype="float32", device=DEVICE)
sd.wait()

print("\nRecording finished.")
sf.write(OUTPUT_PATH, audio, SAMPLE_RATE)
print("Saved:", OUTPUT_PATH)
