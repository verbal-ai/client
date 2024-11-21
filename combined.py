import os
import argparse
import asyncio
import base64
import json
import sys
import time
import threading
import websockets
import torch
import numpy as np
import sounddevice as sd
import re
import queue
import logging
import subprocess
import enum
from dataclasses import dataclass
from Levenshtein import distance
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosedError

from modules.audio import play_audio
from modules.logging import log_tool_call, log_error, log_info, log_warning, logger, log_ws_event
from modules.tools import function_map, tools
from modules.utils import (
    RUN_TIME_TABLE_LOG_JSON,
    SESSION_INSTRUCTIONS,
    PREFIX_PADDING_MS,
    SILENCE_THRESHOLD,
    SILENCE_DURATION_MS,
)

SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
SPEAKING_THRESHOLD = 0.5
AWAKE_THRESHOLD = 15
WAKE_WORD_SIMILARITY_THRESHOLD = 0.9
WAKE_WORD = "hello"

load_dotenv()


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


def base64_encode_audio(audio_bytes):
    return base64.b64encode(audio_bytes).decode("utf-8")


class AsyncMicrophone:
    def __init__(self):
        self.is_recording = False
        self.is_receiving = False
        self.audio_data = []
        self.lock = threading.Lock()

    def start_recording(self):
        with self.lock:
            self.is_recording = True
            self.audio_data = []

    def stop_recording(self):
        with self.lock:
            self.is_recording = False

    def start_receiving(self):
        with self.lock:
            self.is_receiving = True

    def stop_receiving(self):
        with self.lock:
            self.is_receiving = False

    def get_audio_data(self):
        with self.lock:
            data = b''.join(self.audio_data)
            self.audio_data = []
            return data

    def close(self):
        pass


class WakeWordDetector:
    def __init__(self, config: Optional[AudioConfig] = None, mic: Optional[AsyncMicrophone] = None, realtime_api=None):
        self.logger = logger
        self.config = config or AudioConfig()
        self.current_state = State.IDLE
        self.mic = mic or AsyncMicrophone()
        self.realtime_api = realtime_api

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
            if self.mic.is_recording:
                self.mic.audio_data.append(indata.copy().tobytes())
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
        """Start wake word detection."""
        self.logger.info("Starting wake word detection...")

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
                                self.logger.info("Wake word detected! Starting conversation...")
                                if self.realtime_api:
                                    asyncio.run(self.realtime_api.run())
                                break

                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error in detection loop: {e}")

        except KeyboardInterrupt:
            self.logger.info("Wake word detection stopped by user.")
        finally:
            self._stop_whisper_stream()


class RealtimeAPI:
    def __init__(self, prompts=None, mic=None):
        self.prompts = prompts
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Please set the OPENAI_API_KEY in your .env file.")
            sys.exit(1)
        self.exit_event = asyncio.Event()
        self.mic = mic or AsyncMicrophone()

        self.assistant_reply = ""
        self.audio_chunks = []
        self.response_in_progress = False
        self.function_call = None
        self.function_call_args = ""
        self.response_start_time = None

    async def run(self):
        logger.info("Starting realtime conversation...")
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
                    logger.info("✅ Connected to the server.")

                    await self.initialize_session(websocket)
                    ws_task = asyncio.create_task(self.process_ws_messages(websocket))

                    logger.info("Conversation started. Speak freely.")

                    if self.prompts:
                        await self.send_initial_prompts(websocket)
                    else:
                        self.mic.start_recording()
                        logger.info("Recording started. Listening for speech...")

                    await self.send_audio_loop(websocket)

                    await ws_task
                break
            except Exception as e:
                logger.error(f"Conversation error: {e}")
                break
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
        logger.info(f"Received event type: {event_type}")
        logger.debug(f"Full event details: {json.dumps(event, indent=2)}")

        try:
            if event_type == "response.created":
                self.mic.start_receiving()
                self.response_in_progress = True
                logger.info("Response creation started. Stopping recording.")
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
            else:
                logger.info(f"Unhandled event type: {event_type}")
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}")
            traceback.print_exc()

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
                await asyncio.sleep(0.1)
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
                    await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Closing the connection.")
        finally:
            self.exit_event.set()
            self.mic.stop_recording()
            self.mic.close()
            await websocket.close()

    async def send_audio_loop(self, websocket):
        try:
            audio_sent_count = 0
            max_audio_chunks = 50  # Adjust this value as needed

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

                            audio_sent_count += 1

                            # Explicitly commit audio buffer and create response after a certain number of chunks
                            if audio_sent_count >= max_audio_chunks:
                                logger.info(f"Sending audio buffer commit after {audio_sent_count} chunks")

                                # Commit the audio buffer
                                commit_event = {"type": "input_audio_buffer.commit"}
                                log_ws_event("Outgoing", commit_event)
                                await websocket.send(json.dumps(commit_event))

                                # Create a response
                                response_create_event = {"type": "response.create"}
                                log_ws_event("Outgoing", response_create_event)
                                await websocket.send(json.dumps(response_create_event))

                                # Reset the counter
                                audio_sent_count = 0
                    else:
                        logger.debug("No audio data to send")
                else:
                    await asyncio.sleep(0.1)  # Wait while receiving assistant response
                    logger.debug("Currently receiving response, waiting...")

        except Exception as e:
            logger.error(f"Error in send_audio_loop: {e}")
            traceback.print_exc()  # Print full stack trace
        finally:
            self.exit_event.set()
            self.mic.stop_recording()
            self.mic.close()
            await websocket.close()


def main():
    personalization_file = os.getenv("PERSONALIZATION_FILE", "personalization.json")

    prompts = None
    try:
        if os.path.exists(personalization_file):
            with open(personalization_file, "r") as f:
                personalization = json.load(f)
                prompts = personalization.get('initial_prompts', [])
    except Exception as e:
        logger.warning(f"Could not load personalization file: {e}")

    mic = AsyncMicrophone()

    realtime_api = RealtimeAPI(prompts, mic)

    detector = WakeWordDetector(mic=mic, realtime_api=realtime_api)

    print("Wake Word Assistant Started!")
    print(f"Wake Word: '{WAKE_WORD}'")
    print("Press Ctrl+C to exit.")

    try:
        detector.start()
    except KeyboardInterrupt:
        logger.info("Program terminated by user")


if __name__ == "__main__":
    main()
