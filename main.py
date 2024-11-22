import asyncio
from awake import RealtimeAPI
from sleeping import WakeWordDetector


async def main_loop():
    while True:
        detector = WakeWordDetector()
        is_awake = detector.start()

        if is_awake:
            realtime_api_instance = RealtimeAPI(silence_timeout=5)
            should_continue = await realtime_api_instance.run()

            if not should_continue:
                break
        else:
            break

if __name__ == "__main__":
    asyncio.run(main_loop())
