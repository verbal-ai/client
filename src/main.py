from dataclasses import dataclass
from scipy.io.wavfile import write
from typing import Optional, List, Dict, Any, Tuple
from Levenshtein import distance
# from led.setup import green_pin, red_pin, setup_leds
# from led.functions import turn_on_pin, turn_off_pin
import enum
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
WAKE_WORD = "hello"
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
    wake_word = "hello"
    wake_word_similarity_threshold = 0.9


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


history = []


class WhisperStreamProcessor:
    def __init__(self, logger):
        self.logger = logger
        self.buffer = []
        self.wake_word_found = False
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
        self.wake_word_found = False


class AudioProcessor:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.logger = self._setup_logger()
        self.config = config or AudioConfig()
        self.logger.info(f"Initializing AudioProcessor with sample rate: {self.config.sample_rate}")

        self._init_vad_model()
        self.current_state = State.IDLE
        self.last_utterance = time.time() - AWAKE_THRESHOLD
        self.audio_chunks = queue.Queue()
        self.lock = threading.Lock()
        self._audio_data: Optional[np.ndarray] = None

        self.whisper_stream: Optional[subprocess.Popen] = None
        self.whisper_processor: Optional[WhisperStreamProcessor] = None
        self.processing = True

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger(__name__)

    def _start_whisper_stream(self):
        """Initialize and start the Whisper stream process."""
        cmd = ["../modules/whisper.cpp/stream", "-m", "../modules/whisper.cpp/models/ggml-base.en.bin"]
        self.whisper_stream = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Remove universal_newlines=True to keep binary mode
            bufsize=0  # No buffering for binary data
        )
        self.whisper_processor = WhisperStreamProcessor(self.logger)

    def _process_whisper_output(self) -> bool:
        """Non-blocking check of Whisper output"""
        if not self.whisper_stream or not self.whisper_processor:
            return False

        if self.whisper_stream.poll() is not None:
            self.logger.error("Whisper stream process has terminated unexpectedly")
            return False

        try:
            # Non-blocking read from stdout
            import select
            reads, _, _ = select.select([self.whisper_stream.stdout], [], [], 0)
            if not reads:
                return False

            line = self.whisper_stream.stdout.readline()
            if not line:
                return False

            # Decode bytes to string
            line_str = line.decode('utf-8')
            self.logger.debug(f"Raw Whisper output: {line_str.strip()}")
            if self.whisper_processor.process_line(line_str):
                return True

        except Exception as e:
            self.logger.error(f"Error processing Whisper output: {e}")
            return False

        return False

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

    def _process_audio_chunk(self, chunk):
        """Process a single audio chunk through Whisper"""
        if self.whisper_stream and self.whisper_stream.poll() is None:
            try:
                int16_data = float32_to_int16(chunk)
                # Write raw bytes to stdin
                self.whisper_stream.stdin.write(int16_data.tobytes())
                self.whisper_stream.stdin.flush()
            except Exception as e:
                self.logger.error(f"Error writing to Whisper stream: {e}")
                self._stop_whisper_stream()
                self._start_whisper_stream()

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
        try:
            self.audio_chunks.put(indata.copy())
        except queue.Full:
            self.logger.warning("Audio buffer full, dropping chunk")

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
        """Record and process audio"""
        self.logger.info("Starting audio recording...")
        is_speaking = False
        silence_start = None
        accumulated_chunks = []

        try:
            AudioDevice.log_devices(self.logger)
            self._start_whisper_stream()

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
                    try:
                        chunk = self.audio_chunks.get(timeout=0.1)
                        current_chunk = chunk.flatten()

                        tensor = torch.from_numpy(current_chunk).to(self.device)
                        speech_prob = self.model(tensor, self.config.sample_rate).item()

                        if speech_prob >= self.config.speaking_threshold:
                            if not is_speaking:
                                self.logger.info("Voice detected, recording...")
                                is_speaking = True
                            silence_start = None
                            accumulated_chunks.append(current_chunk)
                            self._process_audio_chunk(current_chunk)

                            if self._process_whisper_output():
                                self.logger.info("Wake word detected!")
                                self._audio_data = np.concatenate(accumulated_chunks)
                                return True

                        elif is_speaking:
                            if silence_start is None:
                                silence_start = time.time()
                                accumulated_chunks.append(current_chunk)
                            elif time.time() - silence_start > self.config.silence_threshold:
                                self.logger.info("Silence detected, processing final audio...")
                                self._audio_data = np.concatenate(accumulated_chunks)
                                if time.time() - self.last_utterance < self.config.awake_threshold:
                                    return True
                                if self._process_whisper_output():
                                    return True
                                return False

                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing audio chunk: {e}")
                        return False

        except Exception as e:
            self.logger.error(f"Error in audio recording: {e}")
            return False
        finally:
            self._stop_whisper_stream()

        return False

    def process_audio(self) -> Optional[str]:
        """
        Process the recorded audio data and return the response text.
        """
        if self._audio_data is None:
            self.logger.error("No audio data available for processing")
            return None

        try:
            # turn_on_pin(red_pin)
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
            # turn_off_pin(red_pin)

    def play_audio(self, text: str) -> None:
        """
        Placeholder for text-to-speech implementation.
        Replace with actual TTS implementation.
        """
        try:
            # turn_on_pin(red_pin)
            stream_audio(text)
            self.logger.info(f"Playing response: {text}")
            time.sleep(2)  # Simulating audio playback
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            self.current_state = State.ERROR
        finally:
            print("Turning off red pin")
            # turn_off_pin(red_pin)

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
import os

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
                        # output_device_index=0,  # Headphone device index
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
    # setup_leds()
    processor = AudioProcessor()
    processor.run()


if __name__ == "__main__":
    main()

#######################################################################################################################

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import websockets
from datetime import datetime
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosedError

# Import from modules
from .modules.async_microphone import AsyncMicrophone
from .modules.audio import play_audio
from .modules.logging import log_tool_call, log_error, log_info, log_warning, logger, log_ws_event
from .modules.tools import function_map, tools
from .modules.utils import (
    RUN_TIME_TABLE_LOG_JSON,
    SESSION_INSTRUCTIONS,
    PREFIX_PADDING_MS,
    SILENCE_THRESHOLD,
    SILENCE_DURATION_MS,
)

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = ["OPENAI_API_KEY", "PERSONALIZATION_FILE"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error("Please set these variables in your .env file.")
    sys.exit(1)

# Load personalization data
with open(os.getenv("PERSONALIZATION_FILE"), "r") as f:
    personalization = json.load(f)


def base64_encode_audio(audio_bytes):
    return base64.b64encode(audio_bytes).decode("utf-8")


def log_runtime(function_or_name: str, duration: float):
    jsonl_file = RUN_TIME_TABLE_LOG_JSON
    time_record = {
        "timestamp": datetime.now().isoformat(),
        "function": function_or_name,
        "duration": f"{duration:.4f}",
    }
    with open(jsonl_file, "a") as file:
        json.dump(time_record, file)
        file.write("\n")

    logger.info(f"⏰ {function_or_name}() took {duration:.4f} seconds")


class RealtimeAPI:
    def __init__(self, prompts=None):
        self.prompts = prompts
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Please set the OPENAI_API_KEY in your .env file.")
            sys.exit(1)
        self.exit_event = asyncio.Event()
        self.mic = AsyncMicrophone()

        # Initialize state variables
        self.assistant_reply = ""
        self.audio_chunks = []
        self.response_in_progress = False
        self.function_call = None
        self.function_call_args = ""
        self.response_start_time = None

    async def run(self):
        while True:
            try:
                url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1",
                }

                async with websockets.connect(
                    url,
                    extra_headers=headers,
                    close_timeout=120,
                    ping_interval=60,
                    ping_timeout=10,
                ) as websocket:
                    log_info("✅ Connected to the server.", style="bold green")

                    await self.initialize_session(websocket)
                    ws_task = asyncio.create_task(self.process_ws_messages(websocket))

                    logger.info(
                        "Conversation started. Speak freely, and the assistant will respond."
                    )

                    if self.prompts:
                        await self.send_initial_prompts(websocket)
                    else:
                        self.mic.start_recording()
                        logger.info("Recording started. Listening for speech...")

                    await self.send_audio_loop(websocket)

                    logger.info("before await ws_task")

                    # Wait for the WebSocket processing task to complete
                    await ws_task

                    logger.info("await ws_task complete")

                # If execution reaches here without exceptions, exit the loop
                break
            except ConnectionClosedError as e:
                if "keepalive ping timeout" in str(e):
                    logger.warning(
                        "WebSocket connection lost due to keepalive ping timeout. Reconnecting..."
                    )
                    await asyncio.sleep(1)  # Wait before reconnecting
                    continue  # Retry the connection
                else:
                    logger.exception("WebSocket connection closed unexpectedly.")
                    break  # Exit the loop on other connection errors
            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")
                break  # Exit the loop on unexpected exceptions
            finally:
                self.mic.stop_recording()
                self.mic.close()

    async def initialize_session(self, websocket):
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": SESSION_INSTRUCTIONS,
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": SILENCE_THRESHOLD,
                    "prefix_padding_ms": PREFIX_PADDING_MS,
                    "silence_duration_ms": SILENCE_DURATION_MS,
                },
                "tools": tools,
            },
        }
        log_ws_event("Outgoing", session_update)
        await websocket.send(json.dumps(session_update))

    async def process_ws_messages(self, websocket):
        while True:
            try:
                message = await websocket.recv()
                event = json.loads(message)
                log_ws_event("Incoming", event)
                await self.handle_event(event, websocket)
            except websockets.ConnectionClosed:
                log_warning("⚠️ WebSocket connection lost.")
                break

    async def handle_event(self, event, websocket):
        event_type = event.get("type")
        if event_type == "response.created":
            self.mic.start_receiving()
            self.response_in_progress = True
        elif event_type == "response.output_item.added":
            await self.handle_output_item_added(event)
        elif event_type == "response.function_call_arguments.delta":
            self.function_call_args += event.get("delta", "")
        elif event_type == "response.function_call_arguments.done":
            await self.handle_function_call(event, websocket)
        elif event_type == "response.text.delta":
            delta = event.get("delta", "")
            self.assistant_reply += delta
            print(f"Assistant: {delta}", end="", flush=True)
        elif event_type == "response.audio.delta":
            self.audio_chunks.append(base64.b64decode(event["delta"]))
        elif event_type == "response.done":
            await self.handle_response_done()
        elif event_type == "error":
            await self.handle_error(event, websocket)
        elif event_type == "input_audio_buffer.speech_started":
            logger.info("Speech detected, listening...")
        elif event_type == "input_audio_buffer.speech_stopped":
            await self.handle_speech_stopped(websocket)
        elif event_type == "rate_limits.updated":
            self.response_in_progress = False
            self.mic.is_recording = True
            logger.info("Resumed recording after rate_limits.updated")

    async def handle_output_item_added(self, event):
        item = event.get("item", {})
        if item.get("type") == "function_call":
            self.function_call = item
            self.function_call_args = ""

    async def handle_function_call(self, event, websocket):
        if self.function_call:
            function_name = self.function_call.get("name")
            call_id = self.function_call.get("call_id")
            logger.info(
                f"Function call: {function_name} with args: {self.function_call_args}"
            )
            try:
                args = (
                    json.loads(self.function_call_args)
                    if self.function_call_args
                    else {}
                )
            except json.JSONDecodeError:
                args = {}
            await self.execute_function_call(function_name, call_id, args, websocket)

    async def execute_function_call(self, function_name, call_id, args, websocket):
        if function_name in function_map:
            try:
                result = await function_map[function_name](**args)
                log_tool_call(function_name, args, result)
            except Exception as e:
                error_message = f"Error executing function '{function_name}': {str(e)}"
                log_error(error_message)
                result = {"error": error_message}
                await self.send_error_message_to_assistant(error_message, websocket)
        else:
            error_message = f"Function '{function_name}' not found. Add to function_map in tools.py."
            log_error(error_message)
            result = {"error": error_message}
            await self.send_error_message_to_assistant(error_message, websocket)

        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        await websocket.send(json.dumps(function_call_output))
        await websocket.send(json.dumps({"type": "response.create"}))

        # Reset function call state
        self.function_call = None
        self.function_call_args = ""

    async def send_error_message_to_assistant(self, error_message, websocket):
        error_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": error_message}],
            },
        }
        log_ws_event("Outgoing", error_item)
        await websocket.send(json.dumps(error_item))

    async def handle_response_done(self):
        if self.response_start_time is not None:
            response_end_time = time.perf_counter()
            response_duration = response_end_time - self.response_start_time
            log_runtime("realtime_api_response", response_duration)
            self.response_start_time = None

        log_info("Assistant response complete.", style="bold blue")
        if self.audio_chunks:
            audio_data = b"".join(self.audio_chunks)
            logger.info(
                f"Sending {len(audio_data)} bytes of audio data to play_audio()"
            )
            await play_audio(audio_data)
            logger.info("Finished play_audio()")
        self.assistant_reply = ""
        self.audio_chunks = []
        logger.info("Calling stop_receiving()")
        self.mic.stop_receiving()

    async def handle_error(self, event, websocket):
        error_message = event.get("error", {}).get("message", "")
        log_error(f"Error: {error_message}")
        if "buffer is empty" in error_message:
            logger.info("Received 'buffer is empty' error, no audio data sent.")
        elif "Conversation already has an active response" in error_message:
            logger.info("Received 'active response' error, adjusting response flow.")
            self.response_in_progress = True
        else:
            logger.error(f"Unhandled error: {error_message}")

    async def handle_speech_stopped(self, websocket):
        self.mic.stop_recording()
        logger.info("Speech ended, processing...")
        self.response_start_time = time.perf_counter()
        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def send_initial_prompts(self, websocket):
        logger.info(f"Sending {len(self.prompts)} prompts: {self.prompts}")
        content = [{"type": "input_text", "text": prompt} for prompt in self.prompts]
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": content,
            },
        }
        log_ws_event("Outgoing", event)
        await websocket.send(json.dumps(event))

        # Trigger the assistant's response
        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        await websocket.send(json.dumps(response_create_event))

    async def send_audio_loop(self, websocket):
        try:
            while not self.exit_event.is_set():
                await asyncio.sleep(0.1)  # Small delay to accumulate audio data
                if not self.mic.is_receiving:
                    audio_data = self.mic.get_audio_data()
                    if audio_data and len(audio_data) > 0:
                        base64_audio = base64_encode_audio(audio_data)
                        if base64_audio:
                            audio_event = {
                                "type": "input_audio_buffer.append",
                                "audio": base64_audio,
                            }
                            log_ws_event("Outgoing", audio_event)
                            await websocket.send(json.dumps(audio_event))
                        else:
                            logger.debug("No audio data to send")
                else:
                    await asyncio.sleep(0.1)  # Wait while receiving assistant response
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Closing the connection.")
        finally:
            self.exit_event.set()
            self.mic.stop_recording()
            self.mic.close()
            await websocket.close()


def main():
    print(f"Starting realtime API...")
    logger.info(f"Starting realtime API...")
    parser = argparse.ArgumentParser(
        description="Run the realtime API with optional prompts."
    )
    parser.add_argument("--prompts", type=str, help="Prompts separated by |")
    args = parser.parse_args()

    prompts = args.prompts.split("|") if args.prompts else None

    realtime_api_instance = RealtimeAPI(prompts)
    try:
        asyncio.run(realtime_api_instance.run())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    print("Press Ctrl+C to exit the program.")
    main()
