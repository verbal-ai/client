import asyncio
import websockets

async def test_websocket():
    # Echo test server
    uri = "wss://echo.websocket.org"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to echo server")
            await websocket.send("Hello, World!")
            response = await websocket.recv()
            print(f"Received: {response}")
            
    except Exception as e:
        print(f"Error: {type(e).__name__} - {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())