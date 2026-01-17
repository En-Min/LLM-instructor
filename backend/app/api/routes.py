from fastapi import APIRouter

from app.db.crud import create_session
from app.db.database import init_db
from app.execution.test_runner import run_test

router = APIRouter()


@router.get("/api/sessions/create")
def create_new_session():
    init_db()
    session = create_session()
    return {"session_id": session.id}


@router.get("/api/progress/{session_id}")
def get_session_progress(session_id: int):
    return []


@router.post("/api/test/run")
def run_function_test(function_name: str):
    return run_test(function_name)
