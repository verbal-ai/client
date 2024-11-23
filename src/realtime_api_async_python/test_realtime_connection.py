import asyncio
import websockets
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def test_realtime_api_connection():
    # Retrieve your API key from the environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable in your .env file.")
        return

    # Define the WebSocket URL with the appropriate model
    # Replace 'gpt-4' with the correct model name if necessary
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

    # Set the required headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    # Print request details (safely)
    print("\nRequest Details:")
    print(f"URL: {url}")
    print("Headers:")
    for key, value in headers.items():
        if key == "Authorization":
            # Show only first/last 4 chars of API key
            masked_value = f"Bearer {value[7:11]}...{value[-4:]}"
            print(f"  {key}: {masked_value}")
        else:
            print(f"  {key}: {value}")

    # Attempt to establish the WebSocket connection
    try:
        async with websockets.connect(
            url, 
            additional_headers=headers,
            open_timeout=30
        ) as websocket:
            print("\nConnected to the server.")
    except websockets.InvalidStatusCode as e:
        print(f"\nFailed to connect: {e}")
        if e.status_code == 403:
            print("HTTP 403 Forbidden: Access denied.")
            print("You may not have access to the Realtime API.")
        else:
            print(f"HTTP {e.status_code}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__} - {e}")


if __name__ == "__main__":
    asyncio.run(test_realtime_api_connection())
