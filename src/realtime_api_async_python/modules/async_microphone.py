import pyaudio
import queue
import logging
import numpy as np
from scipy import signal
from .utils import FORMAT, CHANNELS, RATE, CHUNK

class AsyncMicrophone:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        
        # Print available input devices
        # logging.info("\nAvailable Audio Input Devices:")
        # for i in range(self.p.get_device_count()):
        #     dev_info = self.p.get_device_info_by_index(i)
        #     if dev_info['maxInputChannels'] > 0:  # Only show input devices
        #         logging.info(f"Device {i}: {dev_info['name']}")
        
        # Get default input device
        default_input = self.p.get_default_input_device_info()
        logging.info(f"\nUsing input device: {default_input['name']}")
        
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback,
        )
        
        # Log audio configuration
        logging.info(f"Audio Configuration:")
        logging.info(f"Format: {FORMAT}")
        logging.info(f"Channels: {CHANNELS}")
        logging.info(f"Sample Rate: {RATE} Hz")
        logging.info(f"Chunk Size: {CHUNK}")
        
        self.queue = queue.Queue()
        self.is_recording = False
        self.is_receiving = False
        self.target_rate = 24000
        logging.info(f"Resampling to {self.target_rate} Hz")

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording and not self.is_receiving:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Apply a low-pass filter before downsampling to prevent aliasing
            nyq = self.target_rate / 2
            cutoff = nyq * 0.9  # Cut off at 90% of Nyquist frequency
            b, a = signal.butter(4, cutoff / (RATE / 2), btype='low')
            filtered = signal.filtfilt(b, a, audio_data)

            # Resample using scipy.signal.resample_poly with anti-aliasing
            resampled = signal.resample_poly(filtered, up=self.target_rate, down=RATE, window=('kaiser', 5.0))

            # Convert back to int16
            resampled_16int = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)
            self.queue.put(resampled_16int.tobytes())

        return (None, pyaudio.paContinue)

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

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        logging.info("AsyncMicrophone closed")
