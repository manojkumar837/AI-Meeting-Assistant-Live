"""Finds a Realtek microphone specifically and tests it in mono mode."""

import numpy as np
import sounddevice as sd

SECONDS = 5

print("=== AVAILABLE INPUT DEVICES ===")
realtek_device = None
for i, device in enumerate(sd.query_devices()):
    if device["max_input_channels"] <= 0:
        continue

    print(f"{i}: {device['name']} | channels={device['max_input_channels']} | "
          f"rate={device['default_samplerate']}")

    name = device["name"].lower()
    if "microphone" in name and "realtek" in name and realtek_device is None:
        realtek_device = i

print("\n================================")
if realtek_device is None:
    print("ERROR: Realtek microphone was not found.")
    raise SystemExit

info = sd.query_devices(realtek_device, "input")
print("Selected device:", realtek_device)
print("Name:", info["name"])
print("Input channels:", info["max_input_channels"])
print("Default sample rate:", info["default_samplerate"])

sample_rate = int(info["default_samplerate"])
print("\nTesting microphone...")
print("Channels: 1")
print("Sample rate:", sample_rate)

try:
    sd.check_input_settings(device=realtek_device, samplerate=sample_rate,
                             channels=1, dtype="float32")
    print("\nCHECK SUCCESS")

    stream = sd.InputStream(device=realtek_device, samplerate=sample_rate,
                             channels=1, dtype="float32", blocksize=sample_rate)
    stream.start()
    print("MICROPHONE OPENED SUCCESSFULLY")
    print(f"Speak now for {SECONDS} seconds...\n")

    for i in range(SECONDS):
        data, overflowed = stream.read(sample_rate)
        peak = float(np.max(np.abs(data)))
        print(f"Second {i + 1}: shape={data.shape}, peak={peak:.6f}, overflow={overflowed}")

    stream.stop()
    stream.close()
    print("\nMICROPHONE TEST PASSED")

except Exception as e:
    print("\nMICROPHONE TEST FAILED")
    print(type(e).__name__)
    print(e)
