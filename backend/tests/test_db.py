from app.db.database import init_db
from app.db.crud import create_session, update_progress, get_progress


def test_progress_crud(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("DATABASE_URL", db_url)

    init_db()
    session = create_session()
    assert session.id is not None

    update_progress(session.id, "linear", state="concept_check")
    progress = get_progress(session.id, "linear")
    assert progress is not None
    assert progress.state == "concept_check"
