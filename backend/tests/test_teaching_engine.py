from app.agent.teaching_engine import TeachingEngine, TeachingState
from app.db.database import init_db
from app.db.crud import create_session


def test_state_transitions(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    init_db()
    session = create_session()

    engine = TeachingEngine(session.id)
    assert engine.get_state("linear") == TeachingState.CONCEPT_CHECK

    engine.transition("linear", "understood", {})
    assert engine.get_state("linear") == TeachingState.IMPLEMENTATION
