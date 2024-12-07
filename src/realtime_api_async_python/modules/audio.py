import asyncio
import pyaudio
import numpy as np
from .utils import FORMAT, CHANNELS, RATE
from .logging import logger, log_info, log_warning, log_error

async def play_audio(audio_data, device_index=None, volume_multiplier=4.0):
    p = pyaudio.PyAudio()
    
    try:
        # Convert bytes to numpy array for volume adjustment
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Increase volume (multiply the amplitude)
        audio_array = audio_array * volume_multiplier
        
        # Prevent clipping by clipping to 16-bit integer range
        audio_array = np.clip(audio_array, -32768, 32767)
        
        # Convert back to bytes
        amplified_audio = audio_array.astype(np.int16).tobytes()
        
        # Open stream with specific device
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=24000,
            output=True,
            output_device_index=0  # 3.5mm jack
        )
        
        current_device = p.get_device_info_by_index(0)
        log_info(f"Using audio device: {current_device['name']}", style="bold green")
        log_info(f"Volume multiplier: {volume_multiplier}x", style="bold blue")
        
        # Play amplified audio
        stream.write(amplified_audio)

        # Add silence at the end
        silence_duration = 0.4
        silence_frames = int(RATE * silence_duration)
        silence = b"\x00" * (silence_frames * CHANNELS * 2)
        stream.write(silence)

        await asyncio.sleep(0.5)

    except Exception as e:
        log_error(f"Audio playback error: {str(e)}")
        raise
    
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        log_info("Audio playback completed", style="bold blue")
