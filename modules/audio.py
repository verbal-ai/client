import asyncio
import pyaudio
import logging
from contextlib import asynccontextmanager
from .utils import FORMAT, CHANNELS, RATE


@asynccontextmanager
async def create_audio_player():
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
        yield stream
    finally:
        if stream:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        p.terminate()


async def play_audio(audio_data):
    try:
        async with create_audio_player() as stream:
            stream.write(audio_data)

            silence_duration = 0.4
            silence_frames = int(RATE * silence_duration)
            silence = b"\x00" * (silence_frames * CHANNELS * 2)
            stream.write(silence)

            await asyncio.sleep(0.5)

        logging.debug("Audio playback completed successfully")
    except Exception as e:
        logging.error(f"Error during audio playback: {e}")
        raise
