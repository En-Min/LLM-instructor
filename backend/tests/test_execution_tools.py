from app.execution.code_reader import read_student_code


def test_read_student_code_rejects_outside(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSIGNMENT_DIR", str(tmp_path))
    result = read_student_code("../outside.txt")
    assert "outside" in result.lower()
