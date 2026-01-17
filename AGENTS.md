# Repository Guidelines

## Project Structure & Module Organization

- Root contains reference materials only: `assignment1-basics-main/`, `assignment2-systems-main/`, `assignment3-scaling-main/`, `assignment4-data-main/`, `assignment5-alignment-main/`, and `spring2025-lectures-main/`.
- Each assignment has its own nested repo folder (e.g., `assignment1-basics-main/assignment1-basics-main/`) with `README.md`, `pyproject.toml`, and typically `tests/` for context.
- The active project guidance lives at the root in `SPEC.md` (implementation plan) and `TODO.md` (current task list); keep those updated as work progresses.

## Build, Test, and Development Commands

- Python assignments use `uv` for environments and deps: `uv run python` (drops into a REPL with deps installed).
- Run tests from an assignment repo: `uv run pytest`.
- Assignment 2 submission helper: `./test_and_make_submission.sh` (runs tests and bundles a submission tarball).
- Lectures: `python execute.py -m lecture_01` (builds a trace into `var/traces/`).
- Trace viewer (lecture frontend): `npm run dev` inside `spring2025-lectures-main/spring2025-lectures-main/trace-viewer/`.

## Coding Style & Naming Conventions

- Python is the primary language; follow `pyproject.toml` tool settings in each assignment repo.
- Ruff is configured with `line-length = 120` in assignments; keep imports and formatting consistent with that.
- Module naming follows `cs336_basics`, `cs336_systems`, etc.; new packages should mirror that snake_case pattern.

## Testing Guidelines

- Tests are in `tests/` directories inside each assignment repo and run with `pytest`.
- Prefer naming tests as `test_*.py` and functions as `test_*` to match pytest discovery.
- Some assignments expect adapters or starter stubs to be filled in (see each assignment `README.md`).

## Commit & Pull Request Guidelines

- This workspace snapshot does not include Git history, so no local commit convention is visible.
- Use clear, imperative commit subjects (e.g., "Add data loader for TinyStories") and include a short PR description with the assignment and what changed.
- If your change affects outputs (plots, traces, or artifacts), link or attach the relevant file and mention how to regenerate it.

## Data & Environment Notes

- Some assignments require downloading datasets into a local `data/` directory; follow the per-assignment `README.md` instructions before running tests.
- Large artifacts should stay out of versioned folders unless explicitly required by an assignment.
