import pyaudio
import wave

def record_audio(seconds=5, filename="output.wav"):
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    # Record audio
    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def play_audio(filename="output.wav"):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the sound file 
    wf = wave.open(filename, 'rb')

    # Open stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                   channels=wf.getnchannels(),
                   rate=wf.getframerate(),
                   output=True)

    # Read data in chunks
    data = wf.readframes(1024)

    # Play the sound by writing the audio data to the stream
    print("* playing")
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    print("* done playing")

    # Close and terminate everything properly
    stream.close()
    p.terminate()

# Example usage
if __name__ == "__main__":
    # Record for 5 seconds
    record_audio(5)
    # Play back the recording
    play_audio()
