"""Opens a specific input device directly and reads 5 seconds of audio,
printing the peak level for each second so you can confirm it's picking
up sound."""

import numpy as np
import sounddevice as sd

DEVICE = 2
SAMPLE_RATE = 44100
CHANNELS = 2
SECONDS = 5

print("Available input devices:\n")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"{i}: {d['name']} | inputs={d['max_input_channels']} | rate={d['default_samplerate']}")

print(f"\n--- Testing device {DEVICE} ---")
info = sd.query_devices(DEVICE)
print("Device:", info["name"])
print("Input channels:", info["max_input_channels"])
print("Default rate:", info["default_samplerate"])

try:
    print("\nOpening microphone...")
    stream = sd.InputStream(
        device=DEVICE, samplerate=SAMPLE_RATE, channels=CHANNELS,
        dtype="float32", blocksize=SAMPLE_RATE,
    )
    stream.start()
    print("SUCCESS: Microphone opened!")
    print(f"Speak for {SECONDS} seconds...\n")

    for i in range(SECONDS):
        data, overflowed = stream.read(SAMPLE_RATE)
        peak = float(np.max(np.abs(data)))
        print(f"Second {i + 1}: shape={data.shape}, peak={peak:.6f}, overflow={overflowed}")

    stream.stop()
    stream.close()
    print("\nMicrophone test completed successfully.")

except Exception as e:
    print("\nMICROPHONE TEST FAILED")
    print(type(e).__name__)
    print(e)
