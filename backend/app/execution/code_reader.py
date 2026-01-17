import os
from pathlib import Path


def read_student_code(file_path: str) -> str:
    assignment_dir = os.getenv("ASSIGNMENT_DIR")
    if not assignment_dir:
        return "Error: ASSIGNMENT_DIR is not set"

    base_path = Path(assignment_dir).resolve()
    full_path = (base_path / file_path).resolve()

    if not str(full_path).startswith(str(base_path)):
        return "Error: Path outside assignment directory"

    if not full_path.exists():
        return f"Error: File not found: {file_path}"

    return full_path.read_text()
