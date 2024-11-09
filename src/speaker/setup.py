import pyaudio
import wave
import time
import sys
import RPi.GPIO as GPIO

speaker_pin = 18
chunk = 1024
p = pyaudio.PyAudio()
pwm = GPIO.PWM(speaker_pin, 440)


def setup_speaker():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(speaker_pin, GPIO.OUT)


def play_file(file_name):
    try:
        wf = wave.open(file_name, 'rb')

        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()

        print(f"Playing: {file_name}")
        print(f"Sample Width: {sample_width} bytes")
        print(f"Frame Rate: {frame_rate} Hz")

        sample_duration = 1.0 / frame_rate

        pwm.start(50)

        data = wf.readframes(chunk)
        while data:
            # Convert byte data to amplitude values
            for i in range(0, len(data), sample_width):
                if sample_width == 1:  # 8-bit audio
                    amplitude = data[i] - 128  # Center around 0
                elif sample_width == 2:  # 16-bit audio
                    amplitude = int.from_bytes(data[i:i+2], 'little', signed=True)
                else:
                    raise Error("Audio isn't 8-bit or 16-bit")

                # Map amplitude to duty cycle (0-100%)
                duty_cycle = max(0, min(100, (amplitude + 32768) / 65536 * 100))
                pwm.ChangeDutyCycle(duty_cycle)

                time.sleep(sample_duration)  # Wait for the duration of the sample

            data = wf.readframes(chunk)

        pwm.stop()  # Stop PWM after playback
        wf.close()

    except FileNotFoundError:
        print(f"Error: File {file_name} not found")
    except wave.Error:
        print("Error: Invalid or corrupted WAV file")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def cleanup():
    pwm.stop()
    GPIO.cleanup()


if __name__ == "__main__":
    try:
        play_file("../output.wav")
    finally:
        cleanup()
