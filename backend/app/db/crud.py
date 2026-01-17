from typing import Optional

from app.db.database import get_session_local
from app.db.models import Message, Progress, Session


def create_session(assignment: str = "assignment1") -> Session:
    db = get_session_local()()
    session = Session(assignment=assignment)
    db.add(session)
    db.commit()
    db.refresh(session)
    db.close()
    return session


def get_progress(session_id: int, function_name: str) -> Optional[Progress]:
    db = get_session_local()()
    progress = (
        db.query(Progress)
        .filter_by(session_id=session_id, function_name=function_name)
        .first()
    )
    db.close()
    return progress


def update_progress(session_id: int, function_name: str, state: Optional[str] = None, **kwargs) -> None:
    db = get_session_local()()
    progress = (
        db.query(Progress)
        .filter_by(session_id=session_id, function_name=function_name)
        .first()
    )
    if not progress:
        progress = Progress(session_id=session_id, function_name=function_name)
        db.add(progress)
    if state:
        progress.state = state
    for key, value in kwargs.items():
        setattr(progress, key, value)
    db.commit()
    db.close()


def save_message(session_id: int, role: str, content: str, llm_used: Optional[str] = None) -> None:
    db = get_session_local()()
    message = Message(session_id=session_id, role=role, content=content, llm_used=llm_used)
    db.add(message)
    db.commit()
    db.close()
