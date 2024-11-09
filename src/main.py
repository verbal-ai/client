import enum
import os
import time
import threading
import sounddevice as sd
import numpy as np
import torch
import base64  # Add this import
import requests
import json
from typing import Optional
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Only console output
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    OUTPUT = "output"
    ERROR = "error"

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        logger.info("Initializing AudioProcessor with sample rate: %d", sample_rate)
        
        # Load VAD model
        model_path = os.path.join(os.path.dirname(__file__), "..", "data", "silero_vad.jit")
        logger.debug("Loading VAD model from: %s", model_path)
        self.model = torch.jit.load(model_path)
        
        # Setup device and model
        self.sample_rate = sample_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Using device: %s", self.device)
        self.model.to(self.device)
        
        # VAD parameters
        self.window_size_samples = 512
        self.speaking_threshold = 0.5
        self.silence_threshold = 3.0
        logger.debug("VAD parameters - window_size: %d, speaking_threshold: %.2f, silence_threshold: %.2f",
                    self.window_size_samples, self.speaking_threshold, self.silence_threshold)
        
        # State management
        self.current_state = State.IDLE
        self.audio_chunks = []
        self.lock = threading.Lock()
        logger.info("AudioProcessor initialization complete")

    def _record_audio(self) -> Optional[np.ndarray]:
        self.audio_chunks = []
        is_speaking = False
        silence_start = None
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Status: {status}")
            self.audio_chunks.append(indata.copy())
        
        try:
            # Log available audio devices
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                logger.info(f"Device {i}: {device['name']}")
            logger.info(f"Using default input device: {default_input['name']}")
            logger.info(f"Default device specs - channels: {default_input['max_input_channels']}, "
                       f"default samplerate: {default_input['default_samplerate']}")

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=self.window_size_samples
            ) as stream:
                logger.info(f"Stream opened - Device: {stream.device}, "
                          f"Samplerate: {stream.samplerate}, "
                          f"Channels: {stream.channels}")
                
                while self.current_state == State.LISTENING:
                    if len(self.audio_chunks) > 0:
                        current_chunk = self.audio_chunks[-1].flatten()
                        tensor = torch.from_numpy(current_chunk).to(self.device)
                        speech_prob = self.model(tensor, self.sample_rate).item()
                        # logger.debug(f"Speech probability: {speech_prob}")
                        
                        if speech_prob >= self.speaking_threshold:
                            if not is_speaking:
                                logger.info("Voice detected, recording...")
                                is_speaking = True
                            silence_start = None
                        elif is_speaking:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > self.silence_threshold:
                                logger.info("Processing audio...")
                                return np.concatenate([chunk.flatten() for chunk in self.audio_chunks])
                        
                    time.sleep(0.01)
        except Exception as e:
            logger.error(f"Recording error: {e}", exc_info=True)
            self.current_state = State.ERROR
            return None
        
        return None
    
    def get_output_device(self):
        """Get the default audio output device"""
        try:
            devices = sd.query_devices()
            default_output = sd.query_devices(kind='output')
            logger.info("Available output devices:")
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    logger.info(f"Device {i}: {device['name']}")
            logger.info(f"Using default output device: {default_output['name']}")
            logger.info(f"Default output specs - channels: {default_output['max_output_channels']}, "
                       f"default samplerate: {default_output['default_samplerate']}")
            return default_output
        except Exception as e:
            logger.error(f"Error getting output device: {e}", exc_info=True)
            return None
        
    
    def play_audio(self,audio_data: np.ndarray):
        """Play audio response"""
        try:
            sd = self.get_output_device()
            sd.play(audio_data, self.sample_rate)
        except Exception as e:
            logger.error(f"Playback error: {e}")
            self.current_state = State.ERROR

    def process_audio(self, audio_data: np.ndarray):
        """Send audio to API and get response"""
        try:
            logger.info("Sending audio data to API (length: %d samples)", len(audio_data))
            
            headers = {
                'content-type': 'application/json'
            }
            
            # Convert to raw binary data
            raw_audio = audio_data.tobytes()
            
            with open('debug_output.txt', 'w') as f:
                f.write(f"Sending data: {json.dumps({
                    'base64Audio': base64.b64encode(raw_audio).decode('utf-8')
                })}")

            logger.debug("Wrote debug data to debug_output.txt")


            self.play_audio(raw_audio)

            response = requests.request(
                "POST",
                "https://us-central1-dictationdaddy.cloudfunctions.net/verbalDemo",
                data=json.dumps({
                    "base64Audio": base64.b64encode(raw_audio).decode('utf-8')
                }),
                headers=headers
            )
            
            logger.debug(f"API Response status: {response.status_code}")
            logger.debug(f"API Response content: {response.text}")
            
            json_response = response.json()
            logger.debug(f"Parsed JSON response: {json_response}")
            
            if 'text' not in json_response:
                logger.error(f"Expected 'text' key in response but got keys: {json_response.keys()}")
                return None
                
            return json_response["text"]
        except Exception as e:
            logger.error(f"API error: {e}", exc_info=True)
            self.current_state = State.ERROR
            return None

    def play_audio(self, text: str):
        """Play audio response"""
        try:
            # TODO: Replace with your text-to-speech implementation
            print(f"Playing response: {text}")
            # Simulating audio playback
            time.sleep(2)
        except Exception as e:
            print(f"Playback error: {e}")
            self.current_state = State.ERROR

    def run(self):
        print("Starting audio processor...")
        
        while True:
            try:
                with self.lock:
                    if self.current_state == State.IDLE:
                        print("Ready to listen...")
                        self.current_state = State.LISTENING
                    
                    elif self.current_state == State.LISTENING:
                        audio_data = self._record_audio()
                        if audio_data is not None:
                            self.current_state = State.THINKING
                    
                    elif self.current_state == State.THINKING:
                        print("Processing your input...")
                        response = self.process_audio(audio_data)
                        if response:
                            self.current_state = State.OUTPUT
                        else:
                            self.current_state = State.ERROR
                    
                    elif self.current_state == State.OUTPUT:
                        self.play_audio(response)
                        self.current_state = State.IDLE
                    
                    elif self.current_state == State.ERROR:
                        print("An error occurred. Resetting...")
                        time.sleep(2)
                        self.current_state = State.IDLE
                
                time.sleep(0.1)  # Prevent CPU overuse
                
            except KeyboardInterrupt:
                print("\nStopping audio processor...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                self.current_state = State.ERROR

def main():
    processor = AudioProcessor()
    processor.run()

if __name__ == "__main__":
    main() 