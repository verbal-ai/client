# Sound Card Configuration on Raspberry Pi

## Initial Device Check

```
aplay -l
```

List of PLAYBACK Hardware Devices
card 0: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
card 1: vc4hdmi1 [vc4-hdmi-1], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
card 2: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]


1. Create/edit ALSA config:

```
sudo nano /etc/asound.conf
```


2. Add configuration:

```
conf
pcm.!default {
type hw
card Headphones # Use the bcm2835 Headphones
}
ctl.!default {
type hw
card Headphones
}
```

## Testing Audio Output

```
speaker-test -D hw:2,0 -c 2 -t sine
```


```
amixer cset numid=3 1  # Force output to 3.5mm jack
```

Restart the ALSA service:

```
sudo /etc/init.d/alsa-utils restart
```

```
speaker-test -D hw:2,0 -c 2 -t sine
```

```python
def audio_player():
    p = pyaudio.PyAudio()
    try:
    # Force use of headphone device (card 2)
    device_index = 2 # bcm2835 Headphones
    print(f"\nUsing Headphone device (index: {device_index})")
    stream = p.open(format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,
    output_device_index=device_index, # Specify the headphone device For us it was 0
    frames_per_buffer=samples_per_chunk)
```
