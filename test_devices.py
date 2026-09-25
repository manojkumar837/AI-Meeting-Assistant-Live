"""Lists all input-capable audio devices and checks one specific device."""

import sounddevice as sd

TEST_DEVICE = 15
TEST_SAMPLE_RATE = 48000
TEST_CHANNELS = 2

print("=== ALL INPUT DEVICES ===")
for i, device in enumerate(sd.query_devices()):
    if device["max_input_channels"] > 0:
        print(f"\nDevice {i}: {device['name']}")
        print("  Input channels:", device["max_input_channels"])
        print("  Default sample rate:", device["default_samplerate"])

print(f"\n=== TEST DEVICE {TEST_DEVICE} ===")
try:
    info = sd.query_devices(TEST_DEVICE, "input")
    print("Name:", info["name"])
    print("Max input channels:", info["max_input_channels"])
    print("Default sample rate:", info["default_samplerate"])

    print("\nChecking input settings...")
    sd.check_input_settings(
        device=TEST_DEVICE, samplerate=TEST_SAMPLE_RATE,
        channels=TEST_CHANNELS, dtype="float32",
    )
    print(f"SUCCESS: {TEST_SAMPLE_RATE} Hz / {TEST_CHANNELS} channels is supported.")

except Exception as e:
    print("FAILED:")
    print(type(e).__name__)
    print(e)
