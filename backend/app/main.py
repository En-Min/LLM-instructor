from fastapi import FastAPI, WebSocket

from app.api.routes import router
from app.api.websocket import chat_handler_async
from app.db.database import init_db

app = FastAPI(title="LLM Instructor Backend")

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: int):
    await chat_handler_async(websocket, session_id)
