from fastapi.testclient import TestClient

from app.main import app


def test_websocket_streams_chunks():
    client = TestClient(app)
    with client.websocket_connect("/ws/chat/1") as ws:
        ws.send_text('{"content":"Hello"}')
        data = ws.receive_json()
        assert data["type"] == "assistant_chunk"
        assert data.get("content") != '{"content":"Hello"}'
