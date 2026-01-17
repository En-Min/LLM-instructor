from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    assignment = Column(String, default="assignment1")

    progress = relationship("Progress", back_populates="session")
    messages = relationship("Message", back_populates="session")


class Progress(Base):
    __tablename__ = "progress"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    function_name = Column(String)
    state = Column(String, default="concept_check")
    attempts = Column(Integer, default=0)
    hints_given = Column(Integer, default=0)
    test_status = Column(String, default="not_run")
    last_error = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    session = relationship("Session", back_populates="progress")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    role = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    llm_used = Column(String, nullable=True)

    session = relationship("Session", back_populates="messages")
