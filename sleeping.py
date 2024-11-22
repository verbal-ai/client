import os
import re
import time
import queue
import logging
import threading
import subprocess
import enum
import torch
import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from Levenshtein import distance
from typing import Optional

SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
SPEAKING_THRESHOLD = 0.5
SILENCE_THRESHOLD = 1
AWAKE_THRESHOLD = 15
WAKE_WORD_SIMILARITY_THRESHOLD = 0.9
WAKE_WORD = "hello"


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
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


def get_word_similarity(word1: str, word2: str) -> float:
    """Calculate similarity between two words using Levenshtein distance."""
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 0
    return 1 - (distance(word1, word2) / max_len)


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    WAKE_WORD_DETECTED = "wake_word_detected"


class WhisperStreamProcessor:
    def __init__(self, logger):
        self.logger = logger
        self.buffer = []
        self.complete_text = ""

    def process_line(self, line: str) -> bool:
        """Process a line of Whisper output and return True if wake word is found."""
        stripped_line = line.strip()
        if not stripped_line:
            return False

        self.logger.debug(f"Processing Whisper line: {stripped_line}")
        self.buffer.append(stripped_line)
        self.complete_text = " ".join(self.buffer)

        normalized_text = normalize_text(self.complete_text)
        words = normalized_text.split()

        for word in words:
            similarity = get_word_similarity(word, WAKE_WORD)
            if similarity >= WAKE_WORD_SIMILARITY_THRESHOLD:
                self.logger.info(f"Wake word detected! Word: '{word}', matched with {WAKE_WORD} "
                                 f"(similarity: {similarity:.2f})")
                return True
        return False

    def clear(self):
        """Clear the buffer and reset state."""
        self.buffer = []
        self.complete_text = ""


@dataclass
class AudioConfig:
    def __init__(self):
        pass

    sample_rate: int = SAMPLE_RATE
    window_size: int = VAD_WINDOW_SIZE
    speaking_threshold: float = SPEAKING_THRESHOLD
    silence_threshold: float = SILENCE_THRESHOLD
    channels: int = 1
    sample_width: int = 2  # 16-bit audio
    wake_word: str = WAKE_WORD
    wake_word_similarity_threshold: float = WAKE_WORD_SIMILARITY_THRESHOLD


class WakeWordDetector:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.logger = _setup_logger()
        self.config = config or AudioConfig()
        self.current_state = State.IDLE

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_vad_model()

        self.audio_chunks = queue.Queue()
        self.whisper_stream: Optional[subprocess.Popen] = None
        self.whisper_processor: Optional[WhisperStreamProcessor] = None
        self.lock = threading.Lock()

    def _init_vad_model(self) -> None:
        model_path = os.path.join(os.path.dirname(__file__), "data", "silero_vad.jit")
        self.logger.debug(f"Loading VAD model from: {model_path}")

        self.logger.info(f"Using device: {self.device}")
        self.model = torch.jit.load(model_path)
        self.model.to(self.device)

    def _start_whisper_stream(self):
        """Initialize and start the Whisper stream process."""
        cmd = ["modules/whisper.cpp/stream", "-m", "modules/whisper.cpp/models/ggml-base.en.bin"]
        self.whisper_stream = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        self.whisper_processor = WhisperStreamProcessor(self.logger)

    def _stop_whisper_stream(self):
        """Clean up Whisper stream resources."""
        if self.whisper_stream:
            self.whisper_stream.terminate()
            try:
                self.whisper_stream.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.whisper_stream.kill()

            if self.whisper_stream.stdout:
                self.whisper_stream.stdout.close()
            if self.whisper_stream.stderr:
                self.whisper_stream.stderr.close()

            self.whisper_stream = None
            self.whisper_processor = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time: object, status: object):
        if status:
            self.logger.warning(f"Status: {status}")
        try:
            self.audio_chunks.put(indata.copy())
        except queue.Full:
            self.logger.warning("Audio buffer full, dropping chunk")

    def _process_audio_chunk(self, chunk):
        """Process a single audio chunk through Whisper"""
        if self.whisper_stream and self.whisper_stream.poll() is None:
            try:
                int16_data = _float32_to_int16(chunk)
                self.whisper_stream.stdin.write(int16_data.tobytes())
                self.whisper_stream.stdin.flush()
            except Exception as e:
                self.logger.error(f"Error writing to Whisper stream: {e}")
                self._stop_whisper_stream()
                self._start_whisper_stream()

    def _process_whisper_output(self) -> bool:
        """Non-blocking check of Whisper output"""
        if not self.whisper_stream or not self.whisper_processor:
            return False

        if self.whisper_stream.poll() is not None:
            self.logger.error("Whisper stream process has terminated unexpectedly")
            return False

        try:
            import select
            reads, _, _ = select.select([self.whisper_stream.stdout], [], [], 0)
            if not reads:
                return False

            line = self.whisper_stream.stdout.readline()
            if not line:
                return False

            line_str = line.decode('utf-8')
            self.logger.debug(f"Raw Whisper output: {line_str.strip()}")
            return self.whisper_processor.process_line(line_str)

        except Exception as e:
            self.logger.error(f"Error processing Whisper output: {e}")
            return False

    def start(self):
        """
        Start wake word detection.

        Returns:
            bool: True if wake word was detected, False if stopped by other means
        """
        self.logger.info("Starting wake word detection...")
        wake_word_detected = False

        try:
            self._start_whisper_stream()

            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self.config.window_size
            ) as stream:
                self.logger.info(f"Stream opened: Rate {stream.samplerate}, Channels {stream.channels}")

                while True:
                    try:
                        chunk = self.audio_chunks.get(timeout=0.1)
                        current_chunk = chunk.flatten()

                        tensor = torch.from_numpy(current_chunk).to(self.device)
                        speech_prob = self.model(tensor, self.config.sample_rate).item()

                        if speech_prob >= self.config.speaking_threshold:
                            self._process_audio_chunk(current_chunk)

                            if self._process_whisper_output():
                                self.current_state = State.WAKE_WORD_DETECTED
                                self.logger.info("Wake word detected! Stopping detection...")
                                wake_word_detected = True
                                break

                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error in detection loop: {e}")
                        break

        except KeyboardInterrupt:
            self.logger.info("Wake word detection stopped by user.")
        finally:
            self._stop_whisper_stream()
        return wake_word_detected


def main():
    detector = WakeWordDetector()
    detector.start()


if __name__ == "__main__":
    main()
