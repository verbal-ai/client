import argparse
import asyncio
import logging
from realtime_api_async_python.main import RealtimeAPI
from realtime_api_async_python.wake_word_detector import WakeWordDetector

logger = logging.getLogger(__name__)

async def run_realtime_api(prompts=None, timeout_callback=None):
    """Run the realtime API instance"""
    realtime_api_instance = RealtimeAPI(prompts, timeout_callback=timeout_callback)
    try:
        await realtime_api_instance.run()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")



wake_word_detector = None

async def wake_word_loop():
    """Main loop that handles wake word detection and API execution"""
    try:
        await wake_word_detector.start_listening()
    except KeyboardInterrupt:
        await wake_word_detector.cancel()
        logger.info("Program terminated by user")

async def on_wake_word():
    logger.info("Wake word detected! Starting realtime API...")
    # Get prompts from command line
    prompts = args.prompts.split("|") if args.prompts else None
    # Pause wake word detection while running API
    await wake_word_detector.pause_listening()
    # Run the API
    await run_realtime_api(prompts, timeout_callback=wake_word_loop)
    # Resume wake word detection after API finishes
    await wake_word_detector.start_listening()


def main():
    logger.info("Starting wake word detection system...")
    parser = argparse.ArgumentParser(
        description="Run the wake word detection system with optional prompts."
    )
    parser.add_argument("--prompts", type=str, help="Prompts separated by |")
    global args  # Make args accessible to on_wake_word
    args = parser.parse_args()

    global wake_word_detector
    wake_word_detector = WakeWordDetector(callback=on_wake_word)
    try:
        asyncio.run(wake_word_loop())
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()