#!/bin/bash
set -euo pipefail

export LLM_MODE=local
export DATABASE_URL=${DATABASE_URL:-sqlite:///instructor.db}

uvicorn app.main:app --reload --port 8000
