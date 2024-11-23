import asyncio
import pyaudio
from .utils import FORMAT, CHANNELS, RATE
from .logging import logger, log_info, log_warning, log_error

async def play_audio(audio_data, device_index=None):
    p = pyaudio.PyAudio()
    
    # Print available devices
    log_info("\nAvailable Audio Devices:", style="bold blue")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        log_info(f"Device {i}: {dev_info['name']}", style="blue")
    
    try:
        # Open stream with specific device if provided
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            output_device_index=0 #3.5mm jack
        )
        
        current_device = p.get_device_info_by_index(
            device_index or p.get_default_output_device_info()['index']
        )
        log_info(f"Using audio device: {current_device['name']}", style="bold green")
        
        # Play audio
        stream.write(audio_data)

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
