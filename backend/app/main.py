from fastapi import FastAPI

app = FastAPI(title="LLM Instructor Backend")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
