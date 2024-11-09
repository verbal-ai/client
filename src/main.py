import enum
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import logging
import wave
import io

import sounddevice as sd
import numpy as np
import torch
import base64
import requests

# Constants
SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
SPEAKING_THRESHOLD = 0.5
SILENCE_THRESHOLD = 3.0
API_ENDPOINT = "https://us-central1-dictationdaddy.cloudfunctions.net/verbalDemo"

class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    OUTPUT = "output"
    ERROR = "error"

@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    window_size: int = VAD_WINDOW_SIZE
    speaking_threshold: float = SPEAKING_THRESHOLD
    silence_threshold: float = SILENCE_THRESHOLD
    channels: int = 1
    sample_width: int = 2  # 16-bit audio

class AudioDevice:
    @staticmethod
    def get_devices() -> Dict[str, Any]:
        devices = sd.query_devices()
        return {
            'input': sd.query_devices(kind='input'),
            'output': sd.query_devices(kind='output'),
            'all': devices
        }

    @staticmethod
    def log_devices(logger: logging.Logger) -> None:
        devices = AudioDevice.get_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(devices['all']):
            logger.info(f"Device {i}: {device['name']}")
        logger.info(f"Default input device: {devices['input']['name']}")
        logger.info(f"Default output device: {devices['output']['name']}")

def float32_to_int16(audio_data: np.ndarray) -> np.ndarray:
    """Convert float32 numpy array to int16."""
    return (audio_data * 32767).astype(np.int16)

def create_wav_data(audio_data: np.ndarray, config: AudioConfig) -> bytes:
    """Convert numpy array to WAV file bytes."""
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(config.channels)
            wav_file.setsampwidth(config.sample_width)
            wav_file.setframerate(config.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        return wav_buffer.getvalue()

class AudioProcessor:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.logger = self._setup_logger()
        self.config = config or AudioConfig()
        self.logger.info(f"Initializing AudioProcessor with sample rate: {self.config.sample_rate}")

        self._init_vad_model()
        self.current_state = State.IDLE
        self.audio_chunks: List[np.ndarray] = []
        self.lock = threading.Lock()
        self._audio_data: Optional[np.ndarray] = None

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger(__name__)

    def _init_vad_model(self) -> None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "data", "silero_vad.jit")
        self.logger.debug(f"Loading VAD model from: {model_path}")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")

        self.model = torch.jit.load(model_path)
        self.model.to(self.device)

    def _audio_callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        if status:
            self.logger.warning(f"Status: {status}")
        self.audio_chunks.append(indata.copy())

    def _prepare_audio_for_api(self, audio_data: np.ndarray) -> Tuple[Dict[str, Any], str]:
        """
        Prepare audio data for API submission, returns (payload, format).
        Tries different formats to ensure compatibility.
        """
        # Convert to int16 for consistent formatting
        int16_data = float32_to_int16(audio_data)

        # Create WAV format
        wav_data = create_wav_data(int16_data, self.config)
        wav_base64 = base64.b64encode(wav_data).decode('utf-8')

        # Also prepare raw PCM format as fallback
        pcm_base64 = base64.b64encode(int16_data.tobytes()).decode('utf-8')

        # Try WAV format first
        payload = {
            "base64Audio": wav_base64,
            "format": "wav",
            "sampleRate": self.config.sample_rate,
            "channels": self.config.channels
        }

        return payload, "wav"

    def _record_audio(self) -> bool:
        """Record audio and return True if audio was successfully captured."""
        self.audio_chunks = []
        is_speaking = False
        silence_start = None

        try:
            AudioDevice.log_devices(self.logger)

            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self.config.window_size
            ) as stream:
                self.logger.info(f"Stream opened - Device: {stream.device}, "
                               f"Samplerate: {stream.samplerate}, "
                               f"Channels: {stream.channels}")

                while self.current_state == State.LISTENING:
                    if len(self.audio_chunks) == 0:
                        time.sleep(0.01)
                        continue

                    current_chunk = self.audio_chunks[-1].flatten()
                    tensor = torch.from_numpy(current_chunk).to(self.device)
                    speech_prob = self.model(tensor, self.config.sample_rate).item()

                    if speech_prob >= self.config.speaking_threshold:
                        if not is_speaking:
                            self.logger.info("Voice detected, recording...")
                            is_speaking = True
                        silence_start = None
                    elif is_speaking:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.config.silence_threshold:
                            self.logger.info("Processing audio...")
                            self._audio_data = np.concatenate([chunk.flatten() for chunk in self.audio_chunks])
                            return True

                    time.sleep(0.01)

        except Exception as e:
            self.logger.error(f"Recording error: {e}", exc_info=True)
            self.current_state = State.ERROR
            return False

        return False

    def process_audio(self) -> Optional[str]:
        """
        Process the recorded audio data and return the response text.
        """
        if self._audio_data is None:
            self.logger.error("No audio data available for processing")
            return None

        try:
            self.logger.info(f"Preparing audio data for API (length: {len(self._audio_data)} samples)")

            payload, audio_format = self._prepare_audio_for_api(self._audio_data)
            self.logger.debug(f"Sending audio in {audio_format} format")

            response = requests.post(
                API_ENDPOINT,
                json=payload,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                timeout=30
            )

            self.logger.debug(f"API Response status: {response.status_code}")
            self.logger.debug(f"API Response headers: {dict(response.headers)}")

            if response.status_code != 200:
                self.logger.error(f"API error response: {response.text}")
                try:
                    error_data = response.json()
                    self.logger.error(f"API error details: {error_data}")
                except:
                    self.logger.error(f"Raw API error response: {response.text}")
                return None

            try:
                response_data = response.json()
                self.logger.debug(f"API Response data: {response_data}")

                if 'text' not in response_data:
                    self.logger.error(f"Expected 'text' key in response but got keys: {response_data.keys()}")
                    return None

                return response_data["text"]

            except requests.exceptions.JSONDecodeError as e:
                self.logger.error(f"Failed to parse API response as JSON: {e}")
                self.logger.error(f"Raw response content: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            self.current_state = State.ERROR
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in process_audio: {e}", exc_info=True)
            self.current_state = State.ERROR
            return None

    def play_audio(self, text: str) -> None:
        """
        Placeholder for text-to-speech implementation.
        Replace with actual TTS implementation.
        """
        try:
            self.logger.info(f"Playing response: {text}")
            time.sleep(2)  # Simulating audio playback
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            self.current_state = State.ERROR

    def run(self) -> None:
        self.logger.info("Starting audio processor...")

        while True:
            try:
                with self.lock:
                    match self.current_state:
                        case State.IDLE:
                            self.logger.info("Ready to listen...")
                            self.current_state = State.LISTENING

                        case State.LISTENING:
                            if self._record_audio():
                                self.current_state = State.THINKING

                        case State.THINKING:
                            self.logger.info("Processing your input...")
                            if response := self.process_audio():
                                self.current_state = State.OUTPUT
                            else:
                                self.current_state = State.ERROR

                        case State.OUTPUT:
                            self.play_audio(response)
                            self._audio_data = None  # Clear the audio data
                            self.current_state = State.IDLE

                        case State.ERROR:
                            self.logger.error("An error occurred. Resetting...")
                            self._audio_data = None  # Clear the audio data
                            time.sleep(2)
                            self.current_state = State.IDLE

                time.sleep(0.1)  # Prevent CPU overuse

            except KeyboardInterrupt:
                self.logger.info("\nStopping audio processor...")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                self.current_state = State.ERROR

def main():
    processor = AudioProcessor()
    processor.run()

if __name__ == "__main__":
    main()
