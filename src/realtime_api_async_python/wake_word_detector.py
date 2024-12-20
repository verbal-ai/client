import asyncio
import json
import os
from vosk import Model, KaldiRecognizer
import pyaudio
from typing import Optional, Callable
from realtime_api_async_python.modules.logging import log_info, log_error, log_warning
import sounddevice as sd
import numpy as np

os.environ['NOAUDIT'] = '1'
# Disable unnecessary audio system checks
os.environ['NOAUDIT'] = '1'  # Disable JACK
os.environ['AUDIODEV'] = 'null'  # Disable OSS
os.environ['PULSE_SERVER'] = 'none'  # Disable PulseAudio checks
os.environ['SDL_AUDIODRIVER'] = 'alsa'  # Force ALSA for SDL
os.environ['ALSA_CARD'] = 'Generic'  # Use generic ALSA device


class WakeWordDetector:
    def __init__(
        self,
        model_path: str = os.getenv("VOSK_MODEL_PATH", "data/vosk-model-small-en-us-0.15"),
        wake_word: str = "hey",
        callback: Optional[Callable] = None,
        sample_rate: int = 48000
    ):
        """Initialize wake word detector with Vosk model.
        
        Args:
            model_path: Path to Vosk model directory
            wake_word: Wake word/phrase to detect
            callback: Optional callback function to call when wake word detected
            sample_rate: Audio sample rate in Hz
        """
        if not os.path.exists(model_path):
            log_error(f"Vosk model not found at {model_path}")
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        self.wake_word = wake_word.lower()
        self.callback = callback
        self.sample_rate = sample_rate
        self.is_listening = False
        self.is_paused = False
        self._cancel_event = asyncio.Event()

        self.chunk_size = 4096
        self.format = pyaudio.paInt16
        self.channels = 1

        
        # Initialize Vosk model once and keep it
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        
        # Initialize PyAudio once
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Pre-configure stream parameters

        print("All the parameters:")
        print(f"format: {self.format}")
        print(f"channels: {self.channels}")
        print(f"rate: {self.sample_rate}")
        print(f"chunk_size: {self.chunk_size}")
        self.stream_config = {
            'format': self.format,
            'channels': self.channels,
            'rate': self.sample_rate,
            'input': True,
            'frames_per_buffer': self.chunk_size
        }

        # Add sound parameters
        self.wake_sound = self.generate_wake_sound()
        self.sleep_sound = self.generate_sleep_sound()

    def generate_wake_sound(self):
        """Generate a pleasant ascending tone"""
        duration = 0.2  # seconds
        t = np.linspace(0, duration, int(44100 * duration), False)
        # Generate ascending frequency from 440Hz to 880Hz
        frequency = np.linspace(440, 880, len(t))
        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
        return tone

    def generate_sleep_sound(self):
        """Generate a pleasant descending tone"""
        duration = 0.2  # seconds
        t = np.linspace(0, duration, int(44100 * duration), False)
        # Generate descending frequency from 880Hz to 440Hz
        frequency = np.linspace(880, 440, len(t))
        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
        return tone

    async def play_sound(self, sound_data):
        """Play a sound asynchronously"""
        try:
            sd.play(sound_data, 44100)
            sd.wait()
        except Exception as e:
            log_error(f"Error playing sound: {e}")

    async def start_listening(self):
        """Start or resume listening for wake word asynchronously."""
        if self.is_listening:
            log_warning("Already listening")
            return

        self._cancel_event.clear()
        
        try:
            if self.is_paused:
                # Reinitialize audio resources when resuming
                self.audio = pyaudio.PyAudio()
                self.stream = self.audio.open(**self.stream_config)
                self.recognizer.Reset()
                log_info("▶️ Resumed wake word detection", style="bold green")
            else:
                # Fresh start
                if not self.audio:
                    self.audio = pyaudio.PyAudio()
                if not self.stream:
                    self.stream = self.audio.open(**self.stream_config)
                log_info("✨ Started wake word detection", style="bold green")
            
            # Add a short delay to ensure the stream is ready
            await asyncio.sleep(0.5)
            
            # Play sleep sound when starting to listen
            await self.play_sound(self.sleep_sound)
            log_info("✨ Started wake word detection", style="bold green")
            
            self.is_listening = True
            self.is_paused = False
            
            while self.is_listening and not self._cancel_event.is_set():
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").lower()
                    words = text.split()
                    if any(word == self.wake_word for word in words):
                        # Play wake sound when wake word is detected
                        await self.play_sound(self.wake_sound)
                        log_info(f"🎯 Wake word detected in: '{text}'", style="bold blue")
                        if self.callback and asyncio.iscoroutinefunction(self.callback):
                            await self.callback()
                        elif self.callback:
                            self.callback()
                
                await asyncio.sleep(0.01)

        except Exception as e:
            log_error(f"Error in wake word detection: {e}")
            raise
        finally:
            if not self.is_paused:  # Only cleanup if not paused
                await self.cleanup()

    async def pause_listening(self):
        """Temporarily pause listening and release mic resources."""
        if not self.is_listening:
            return
            
        self.is_listening = False
        self.is_paused = True
        
        # Properly release mic resources
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Terminate PyAudio to fully release mic
        if self.audio:
            self.audio.terminate()
            self.audio = None
            
        log_info("⏸️ Paused wake word detection and released mic")

    async def stop_listening(self):
        """Fully stop listening and cleanup resources."""
        self.is_listening = False
        self.is_paused = False
        await self.cleanup()
        log_info("⏹️ Stopped wake word detection")

    async def cancel(self):
        """Cancel detection and cleanup."""
        log_info("Cancelling wake word detection...")
        self._cancel_event.set()
        await self.stop_listening()

    async def cleanup(self):
        """Clean up stream but keep other resources."""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        log_info("Cleaned up wake word detector stream")

    async def __del__(self):
        """Full cleanup of all resources."""
        await self.stop_listening()
        if self.stream:
            self.stream.close()
        if self.audio:
            self.audio.terminate()

async def test_wake_word_detector():
    """Test function for wake word detection."""
    async def on_wake_word():
        print("Wake word detected!")

    detector = WakeWordDetector(callback=on_wake_word)
    try:
        await detector.start_listening()
    except KeyboardInterrupt:
        detector.pause_listening()
        print("\nStopped wake word detection")

if __name__ == "__main__":
    asyncio.run(test_wake_word_detector())
