import enum
import os
import time
import threading
import sounddevice as sd
import numpy as np
import torch
import requests
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
        # Load VAD model
        model_path = os.path.join(os.path.dirname(__file__), "..", "data", "silero_vad.jit")
        self.model = torch.jit.load(model_path)
        
        # Setup device and model
        self.sample_rate = sample_rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # VAD parameters
        self.window_size_samples = 512
        self.speaking_threshold = 0.5
        self.silence_threshold = 3.0  # 3 seconds
        
        # State management
        self.current_state = State.IDLE
        self.audio_chunks = []
        self.lock = threading.Lock()

    def _record_audio(self) -> Optional[np.ndarray]:
        self.audio_chunks = []
        is_speaking = False
        silence_start = None
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
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

    def process_audio(self, audio_data: np.ndarray):
        """Send audio to API and get response"""
        try:
            # TODO: Replace with your actual API endpoint
            response = requests.post(
                "http://your-api-endpoint/process-audio",
                data=audio_data.tobytes()
            )
            return response.json()["text"]
        except Exception as e:
            print(f"API error: {e}")
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