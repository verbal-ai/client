import argparse
import asyncio
import logging
from realtime_api_async_python.main import RealtimeAPI

logger = logging.getLogger(__name__)

def main():
    print(f"Starting realtime API...")
    logger.info(f"Starting realtime API...")
    parser = argparse.ArgumentParser(
        description="Run the realtime API with optional prompts."
    )
    parser.add_argument("--prompts", type=str, help="Prompts separated by |")
    args = parser.parse_args()

    prompts = args.prompts.split("|") if args.prompts else None

    realtime_api_instance = RealtimeAPI(prompts)
    try:
        asyncio.run(realtime_api_instance.run())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()