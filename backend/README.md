# LLM Instructor Backend

Local CPU-first backend skeleton for development on macOS.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Run

```bash
export LLM_MODE=local
uvicorn app.main:app --reload --port 8000
```

Or run:

```bash
./scripts/run_local.sh
```

## Tests

```bash
pytest
```
