import os
import subprocess
from pathlib import Path


def run_test(function_name: str) -> dict:
    assignment_dir = os.getenv("ASSIGNMENT_DIR")
    if not assignment_dir:
        return {"passed": False, "error": "ASSIGNMENT_DIR is not set"}

    assignment_path = Path(assignment_dir)
    if not assignment_path.exists():
        return {"passed": False, "error": f"Assignment dir not found: {assignment_dir}"}

    cmd = [
        "uv",
        "run",
        "pytest",
        f"tests/test_nn_utils.py::test_{function_name}",
        "-v",
        "--tb=short",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=assignment_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Test timed out"}

    return {
        "function": function_name,
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
