import asyncio
import json
import websockets


async def main():
    # Connect to the FastAPI WebSocket endpoint.
    uri = "ws://127.0.0.1:8000/ws/chat"

    async with websockets.connect(uri) as websocket:

        print("Connected to WebSocket.")

        # ----------------------------------------------------
        # Turn 1
        # ----------------------------------------------------
        message_1 = "What is the status of application APP001?"

        print("\nCLIENT ->", message_1)

        await websocket.send(message_1)

        response_1 = await websocket.recv()

        print("SERVER ->", response_1)

        # ----------------------------------------------------
        # Turn 2
        # ----------------------------------------------------
        message_2 = (
            "What was the escalation score for that application?"
        )

        print("\nCLIENT ->", message_2)

        await websocket.send(message_2)

        response_2 = await websocket.recv()

        print("SERVER ->", response_2)

        # ----------------------------------------------------
        # Turn 3 - PII masking test
        # ----------------------------------------------------
        message_3 = (
            "Please check APP001. "
            "The candidate phone number is 9876543210."
        )

        print("\nCLIENT ->", message_3)

        await websocket.send(message_3)

        response_3 = await websocket.recv()

        print("SERVER ->", response_3)

        print("\nWebSocket test completed.")


if __name__ == "__main__":
    asyncio.run(main())