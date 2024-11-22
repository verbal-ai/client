import pyaudio
import queue
import logging
from contextlib import contextmanager
from .utils import FORMAT, CHANNELS, RATE, CHUNK


class AsyncMicrophone:
    def __init__(self):
        self._initialize_audio()
        self.queue = queue.Queue()
        self.is_recording = False
        self.is_receiving = False
        logging.info("AsyncMicrophone initialized")

    def _initialize_audio(self):
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.callback,
            )
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to initialize audio: {e}")

    def callback(self, in_data, frame_count, time_info, status):
        if status:  # Check for any errors
            logging.warning(f"Audio callback status: {status}")
        if self.is_recording and not self.is_receiving:
            self.queue.put(in_data)
        return None, pyaudio.paContinue

    def start_recording(self):
        self.is_recording = True
        logging.info("Started recording")

    def stop_recording(self):
        self.is_recording = False
        logging.info("Stopped recording")

    def start_receiving(self):
        self.is_receiving = True
        self.is_recording = False
        logging.info("Started receiving assistant response")

    def stop_receiving(self):
        self.is_receiving = False
        logging.info("Stopped receiving assistant response")

    def get_audio_data(self):
        data = b""
        while not self.queue.empty():
            data += self.queue.get()
        return data if data else None

    def _cleanup(self):
        """Internal method to handle resource cleanup"""
        if hasattr(self, 'stream') and self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if hasattr(self, 'p') and self.p:
            self.p.terminate()
            self.p = None

    def close(self):
        """Public method to safely close all resources"""
        try:
            self._cleanup()
            logging.info("AsyncMicrophone closed successfully")
        except Exception as e:
            logging.error(f"Error during AsyncMicrophone cleanup: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
