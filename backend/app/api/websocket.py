import json

import anyio
from fastapi import WebSocket, WebSocketDisconnect

from app.llm.local_client import generate_stream


def chat_handler(websocket: WebSocket, session_id: int) -> None:
    raise NotImplementedError


async def chat_handler_async(websocket: WebSocket, session_id: int) -> None:
    await websocket.accept()
    try:
        async for message in websocket.iter_text():
            try:
                payload = json.loads(message)
                content = payload.get("content", "")
            except json.JSONDecodeError:
                content = message

            if not content:
                continue

            def _stream() -> None:
                try:
                    for chunk in generate_stream(content, max_tokens=64):
                        if not chunk:
                            continue
                        anyio.from_thread.run(
                            websocket.send_json,
                            {"type": "assistant_chunk", "content": chunk},
                        )
                    anyio.from_thread.run(websocket.send_json, {"type": "assistant_end"})
                except Exception:
                    anyio.from_thread.run(
                        websocket.send_json,
                        {
                            "type": "assistant_chunk",
                            "content": "Error: generation failed",
                        },
                    )
                    anyio.from_thread.run(websocket.send_json, {"type": "assistant_end"})

            await anyio.to_thread.run_sync(_stream)
    except WebSocketDisconnect:
        return
