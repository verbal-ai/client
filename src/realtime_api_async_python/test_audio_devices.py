import pyaudio

p = pyaudio.PyAudio()

print("\nAvailable Input Devices:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info['maxInputChannels'] > 0:  # Only show devices that support input
        print(f"Device {i}: {device_info['name']}")
        print(f"  Input @channels: {device_info['maxInputChannels']}")
        print(f"  Sample rate: {int(device_info['defaultSampleRate'])} Hz")


print("\nDefault Input Device:")
try:
    default_device = p.get_default_input_device_info()
    print(f"Device {default_device['index']}: {default_device['name']}")
    print(f"  Input channels: {default_device['maxInputChannels']}")
    print(f"  Sample rate: {int(default_device['defaultSampleRate'])} Hz")
except IOError as e:
    print("No default input device found")

p.terminate()