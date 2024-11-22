import asyncio
from awake import RealtimeAPI
from sleeping import WakeWordDetector


if __name__ == "__main__":
    detector = WakeWordDetector()
    is_awake = detector.start()
    if is_awake:
        realtime_api_instance = RealtimeAPI()
        asyncio.run(realtime_api_instance.run())
