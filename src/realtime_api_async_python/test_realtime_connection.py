import asyncio
import websockets
import os
import time
import ssl
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def check_dns(hostname):
    """Test DNS resolution time."""
    start = time.time()
    try:
        await asyncio.get_event_loop().getaddrinfo(hostname, 443)
        dns_time = time.time() - start
        print(f"DNS resolution took: {dns_time:.2f} seconds")
        return dns_time
    except Exception as e:
        print(f"DNS resolution failed: {e}")
        return None

async def test_ssl_handshake(hostname):
    """Test SSL handshake time."""
    start = time.time()
    try:
        ctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(hostname, 443, ssl=ctx)
        writer.close()
        await writer.wait_closed()
        ssl_time = time.time() - start
        print(f"SSL handshake took: {ssl_time:.2f} seconds")
        return ssl_time
    except Exception as e:
        print(f"SSL handshake failed: {e}")
        return None

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

    # Create SSL context with optimized settings
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    ssl_context.set_ciphers('ECDHE+AESGCM')

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
    asyncio.run(check_dns("api.openai.com"))
    asyncio.run(test_ssl_handshake("api.openai.com"))
    asyncio.run(test_realtime_api_connection())