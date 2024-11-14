from dataclasses import dataclass
from scipy.io.wavfile import write
from typing import Optional, List, Dict, Any, Tuple
from Levenshtein import distance
from led.setup import green_pin, red_pin, setup_leds
from led.functions import turn_on_pin, turn_off_pin
import enum
import os
import time
import threading
import logging
import wave
import io
import queue
import re
import sounddevice as sd
import numpy as np
import torch
import base64
import requests
import subprocess

# Constants
SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
SPEAKING_THRESHOLD = 0.5
SILENCE_THRESHOLD = 1
AWAKE_THRESHOLD = 15
WAKE_WORD = "hey"
WAKE_WORD_SIMILARITY_THRESHOLD = 0.9
API_ENDPOINT = "https://us-central1-dictationdaddy.cloudfunctions.net/verbalDemo"


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    OUTPUT = "output"
    ERROR = "error"


@dataclass
class AudioConfig:
    def __init__(self):
        pass

    sample_rate: int = SAMPLE_RATE
    window_size: int = VAD_WINDOW_SIZE
    speaking_threshold: float = SPEAKING_THRESHOLD
    silence_threshold: float = SILENCE_THRESHOLD
    awake_threshold: float = AWAKE_THRESHOLD
    channels: int = 1
    sample_width: int = 2  # 16-bit audio


class AudioDevice:
    def __init__(self):
        pass

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


def create_wav_file(int16_data):
    with wave.open('output_file.wav', 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(int16_data.tobytes())


def transcribe_audio_locally(model_path, audio_file):
    result = subprocess.run(
        ['../modules/whisper.cpp/main', '-m', model_path, '-f', audio_file],
        capture_output=True,
        text=True
    )
    return result.stdout


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def get_word_similarity(word1: str, word2: str) -> float:
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 0
    return 1 - (distance(word1, word2) / max_len)


def contains_wake_word(audio_data):
    int16_data = float32_to_int16(audio_data)
    create_wav_file(int16_data)

    model_path = "../modules/whisper.cpp/models/ggml-base.en.bin"
    file_path = "./output_file.wav"
    stt = transcribe_audio_locally(model_path, file_path)

    normalized_text = normalize_text(stt)
    words = normalized_text.split()

    for word in words:
        similarity = get_word_similarity(word, WAKE_WORD)
        if similarity >= WAKE_WORD_SIMILARITY_THRESHOLD:
            print(f"Wake word detected! Word: '{word}', matched with {WAKE_WORD} "
                  f"(similarity: {similarity:.2f})")
            return True

    print(f"No wake word detected in text: '{normalized_text}'")
    return False


history = []


class AudioProcessor:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.logger = self._setup_logger()
        self.config = config or AudioConfig()
        self.logger.info(f"Initializing AudioProcessor with sample rate: {self.config.sample_rate}")

        self._init_vad_model()
        self.current_state = State.IDLE
        self.last_utterance = time.time() - AWAKE_THRESHOLD
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
            "history": history,
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
            turn_on_pin(green_pin)
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
                            print(time.time())
                            print(self.last_utterance)
                            if time.time() - self.last_utterance < self.config.awake_threshold:
                                return True
                            elif contains_wake_word(self._audio_data):
                                return True
                            else:
                                return False

                    time.sleep(0.01)

        except Exception as e:
            self.logger.error(f"Recording error: {e}", exc_info=True)
            self.current_state = State.ERROR
            return False

        finally:
            print("Turning off green pin")
            turn_off_pin(green_pin)

        return False

    def process_audio(self) -> Optional[str]:
        """
        Process the recorded audio data and return the response text.
        """
        if self._audio_data is None:
            self.logger.error("No audio data available for processing")
            return None

        try:
            turn_on_pin(red_pin)
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
                self.logger.error(f"API error response: {response.response}")
                try:
                    error_data = response.json()
                    self.logger.error(f"API error details: {error_data}")
                except:
                    self.logger.error(f"Raw API error response: {response.response}")
                return None

            try:
                response_data = response.json()
                self.logger.debug(f"API Response data: {response_data}")
                global history
                history = response_data["history"]

                if 'response' not in response_data:
                    self.logger.error(f"Expected 'text' key in response but got keys: {response_data.keys()}")
                    return None

                return response_data["response"]

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

        finally:
            self.last_utterance = time.time()
            print("Turning off red pin")
            turn_off_pin(red_pin)

    def play_audio(self, text: str) -> None:
        """
        Placeholder for text-to-speech implementation.
        Replace with actual TTS implementation.
        """
        try:
            turn_on_pin(red_pin)
            stream_audio(text)
            self.logger.info(f"Playing response: {text}")
            time.sleep(2)  # Simulating audio playback
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            self.current_state = State.ERROR
        finally:
            print("Turning off red pin")
            turn_off_pin(red_pin)

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





import requests
import pyaudio
import io
import dotenv

dotenv.load_dotenv()

DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=48000&container=none"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Audio parameters (adjust based on your audio stream)
CHUNK_SIZE = 1024
CHANNELS = 1
RATE = 48000
FORMAT = pyaudio.paInt16


def stream_audio(text):
    audio_buffer = queue.Queue(maxsize=50)
    running = threading.Event()
    running.set()
    min_buffers = 5
    buffer_ready = threading.Event()
    
    # Calculate proper audio chunk size
    bytes_per_sample = pyaudio.get_sample_size(FORMAT)
    samples_per_chunk = 1024
    AUDIO_CHUNK_SIZE = bytes_per_sample * CHANNELS * samples_per_chunk
    
    byte_buffer = bytearray()

    def audio_player():
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       output=True,
                       # output_device_index=0, # Specify the headphone device
                       frames_per_buffer=samples_per_chunk)
        
        print("Waiting for initial buffers to fill...")
        buffer_ready.wait()
        print("Starting playback...")
        
        while running.is_set() or not audio_buffer.empty():
            try:
                chunk = audio_buffer.get(timeout=1)
                stream.write(chunk)
            except queue.Empty:
                continue
        
        print("Finished playing all audio")
        stream.stop_stream()
        stream.close()
        p.terminate()

    player_thread = threading.Thread(target=audio_player)
    player_thread.start()

    try:
        payload = {
            "text": text
        }
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(DEEPGRAM_URL, headers=headers, json=payload, stream=True)
        
        buffer_count = 0
        for chunk in response.iter_content():
            if chunk:
                byte_buffer.extend(chunk)
                
                while len(byte_buffer) >= AUDIO_CHUNK_SIZE:
                    audio_chunk = bytes(byte_buffer[:AUDIO_CHUNK_SIZE])
                    audio_buffer.put(audio_chunk)
                    byte_buffer = byte_buffer[AUDIO_CHUNK_SIZE:]
                    buffer_count += 1
                    
                    if buffer_count == min_buffers and not buffer_ready.is_set():
                        print(f"Initial {min_buffers} buffers filled")
                        buffer_ready.set()
        
        # Handle remaining bytes
        if byte_buffer:
            remaining_bytes = len(byte_buffer)
            padding_needed = AUDIO_CHUNK_SIZE - remaining_bytes
            padded_chunk = bytes(byte_buffer) + b'\x00' * padding_needed
            audio_buffer.put(padded_chunk)
            print(f"Added final chunk with {remaining_bytes} bytes of audio + {padding_needed} bytes of padding")

    finally:
        running.clear()
        player_thread.join()
        print("Stream completed")


def main():
    setup_leds()
    processor = AudioProcessor()
    processor.run()


if __name__ == "__main__":
    main()
