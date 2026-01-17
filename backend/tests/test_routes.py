from fastapi.testclient import TestClient

from app.main import app


def test_create_session_route():
    client = TestClient(app)
    response = client.get("/api/sessions/create")
    assert response.status_code == 200
    assert "session_id" in response.json()
