from fastapi import WebSocket, WebSocketDisconnect


def chat_handler(websocket: WebSocket, session_id: int) -> None:
    raise NotImplementedError


async def chat_handler_async(websocket: WebSocket, session_id: int) -> None:
    await websocket.accept()
    try:
        async for message in websocket.iter_text():
            await websocket.send_json({"type": "assistant_chunk", "content": message})
        await websocket.send_json({"type": "assistant_end"})
    except WebSocketDisconnect:
        return
