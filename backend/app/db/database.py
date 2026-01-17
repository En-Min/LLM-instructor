import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

_ENGINE = None
_SESSION_LOCAL = None


def get_engine():
    global _ENGINE, _SESSION_LOCAL
    if _ENGINE is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///instructor.db")
        _ENGINE = create_engine(database_url)
        _SESSION_LOCAL = sessionmaker(bind=_ENGINE)
    return _ENGINE


def get_session_local():
    if _SESSION_LOCAL is None:
        get_engine()
    return _SESSION_LOCAL


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_db():
    db = get_session_local()()
    try:
        yield db
    finally:
        db.close()
