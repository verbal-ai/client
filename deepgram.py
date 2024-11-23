import os
import re
import logging
import requests
import json
import torch
import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import Optional, Callable
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
SPEAKING_THRESHOLD = 0.5
SILENCE_THRESHOLD = 1
WAKE_WORD_SIMILARITY_THRESHOLD = 0.9
WAKE_WORD = "hello"


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('wake_word_detector.log')
        ]
    )
    return logging.getLogger(__name__)


def _float32_to_int16(audio_data: np.ndarray) -> np.ndarray:
    """Convert float32 numpy array to int16."""
    return (audio_data * 32767).astype(np.int16)


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


@dataclass
class AudioConfig:
    def __init__(self):
        pass

    sample_rate: int = SAMPLE_RATE
    window_size: int = VAD_WINDOW_SIZE
    speaking_threshold: float = SPEAKING_THRESHOLD
    silence_threshold: float = SILENCE_THRESHOLD
    channels: int = 1
    sample_width: int = 2
    wake_word: str = WAKE_WORD
    wake_word_similarity_threshold: float = WAKE_WORD_SIMILARITY_THRESHOLD
    vad_buffer_size: int = 5
    vad_threshold: float = 0.5


class DeepgramClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepgram.com/v1/listen"
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/raw"
        }
        self.logger = logging.getLogger(__name__ + ".DeepgramClient")

    def transcribe_audio(self, audio_data: bytes) -> Optional[dict]:
        """Send audio data to Deepgram and get transcription."""
        params = {
            "model": "nova-2",
            "language": "en",
            "punctuate": "true",
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": 1
        }

        try:
            response = requests.post(
                self.base_url,
                params=params,
                headers=self.headers,
                data=audio_data,
                timeout=5  # Reduced timeout for lower latency
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error in Deepgram API request: {e}")
            return None


class DeepgramWakeWordDetector:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.logger = _setup_logger()
        self.config = config or AudioConfig()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_vad_model()

        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set")

        self.deepgram = DeepgramClient(self.api_key)
        self.speech_buffer = BytesIO()

        # VAD state
        self.is_speaking = False
        self.silence_frames = 0
        self.speaking_frames = 0

    def _init_vad_model(self) -> None:
        """Initialize the Silero VAD model."""
        model_path = os.path.join(os.path.dirname(__file__), "data", "silero_vad.jit")
        self.logger.debug(f"Loading VAD model from: {model_path}")

        self.logger.info(f"Using device: {self.device}")
        self.model = torch.jit.load(model_path)
        self.model.to(self.device)

    def _process_vad(self, audio_chunk: np.ndarray) -> bool:
        """Process audio chunk through VAD and return True if speech is detected."""
        tensor = torch.from_numpy(audio_chunk).to(self.device)
        speech_prob = self.model(tensor, self.config.sample_rate).item()

        if speech_prob >= self.config.speaking_threshold:
            self.speaking_frames += 1
            self.silence_frames = 0
            if not self.is_speaking and self.speaking_frames >= 2:
                self.is_speaking = True
                self.logger.debug(f"Speech started (probability: {speech_prob:.2f})")
        else:
            self.silence_frames += 1
            self.speaking_frames = 0
            if self.is_speaking and self.silence_frames >= 5:
                self.is_speaking = False
                self.logger.debug(f"Speech ended (silence frames: {self.silence_frames})")
                return False

        return self.is_speaking

    def _check_wake_word(self, text: str) -> bool:
        """Check if wake word is present in the text."""
        normalized_text = normalize_text(text)
        self.logger.debug(f"Checking for wake word in normalized text: '{normalized_text}'")

        words = normalized_text.split()
        for word in words:
            if word.lower() == self.config.wake_word.lower():
                self.logger.info(f"Wake word detected: {word}")
                return True
        return False

    def _process_speech_segment(self) -> bool:
        """Process accumulated speech segment with Deepgram. Returns True if wake word detected."""
        if self.speech_buffer.tell() == 0:
            return False

        audio_data = self.speech_buffer.getvalue()
        self.speech_buffer = BytesIO()  # Reset buffer

        try:
            response = self.deepgram.transcribe_audio(audio_data)
            if response and "results" in response:
                transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
                if transcript.strip():
                    self.logger.info(f"Transcription: '{transcript}'")
                    if self._check_wake_word(transcript):
                        return True
        except Exception as e:
            self.logger.error(f"Error processing speech segment: {e}")

        return False

    def start(self) -> bool:
        """
        Start wake word detection and run indefinitely until wake word is detected.
        Returns True when wake word is detected.
        """
        self.logger.info("Starting wake word detection...")

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                blocksize=self.config.window_size
            ) as stream:
                self.logger.info("Listening for wake word...")

                while True:  # Run indefinitely
                    audio_chunk, _ = stream.read(self.config.window_size)
                    current_chunk = audio_chunk.flatten()

                    # Process through VAD
                    is_speech = self._process_vad(current_chunk)

                    # Store speech in buffer
                    if is_speech:
                        audio_data = _float32_to_int16(current_chunk)
                        self.speech_buffer.write(audio_data.tobytes())
                    # Process speech when silence is detected
                    elif not is_speech and self.speech_buffer.tell() > 0:
                        if self._process_speech_segment():
                            return True

        except Exception as e:
            self.logger.error(f"Error in wake word detection: {e}")
            return False

def main():
    detector = DeepgramWakeWordDetector()
    detected = detector.start()
    print(f"Wake word {'detected' if detected else 'not detected'}")


if __name__ == "__main__":
    main()
