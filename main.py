import asyncio
from awake import RealtimeAPI
from deepgram import DeepgramWakeWordDetector


async def main_loop():
    while True:
        detector = DeepgramWakeWordDetector()
        is_awake = detector.start(duration=5.0)

        if is_awake:
            realtime_api_instance = RealtimeAPI(silence_timeout=10)
            should_continue = await realtime_api_instance.run()

            if not should_continue:
                break
        else:
            break

if __name__ == "__main__":
    asyncio.run(main_loop())
