import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.llm.local_client import generate_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chat_handler(websocket: WebSocket, session_id: int) -> None:
    raise NotImplementedError


async def chat_handler_async(websocket: WebSocket, session_id: int) -> None:
    await websocket.accept()
    print(f"[WS] Connected: session {session_id}", flush=True)

    try:
        async for message in websocket.iter_text():
            try:
                payload = json.loads(message)
                content = payload.get("content", "")
            except json.JSONDecodeError:
                content = message

            if not content:
                continue

            print(f"[WS] Received: {content[:50]}...")

            # Generate and stream response (synchronous - tiny model is fast)
            try:
                chunk_count = 0
                for chunk in generate_stream(content, max_tokens=64):
                    if not chunk:
                        continue
                    print(f"[WS] Chunk {chunk_count}: {chunk[:20]}")
                    await websocket.send_json({
                        "type": "assistant_chunk",
                        "content": chunk
                    })
                    chunk_count += 1

                print(f"[WS] Done: sent {chunk_count} chunks")
                await websocket.send_json({"type": "assistant_end"})

            except Exception as e:
                print(f"[WS] Error: {e}")
                await websocket.send_json({
                    "type": "assistant_chunk",
                    "content": f"Error: {str(e)}"
                })
                await websocket.send_json({"type": "assistant_end"})

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: session {session_id}")
    except Exception as e:
        print(f"[WS] Exception: {e}")
