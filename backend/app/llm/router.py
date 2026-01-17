from app.config import settings
from app.llm.local_client import generate as generate_local


def generate(prompt: str, max_tokens: int = 256) -> str:
    if settings.llm_mode.lower() == "local":
        return generate_local(prompt, max_tokens=max_tokens)
    if settings.llm_mode.lower() == "vllm":
        raise RuntimeError("vLLM mode not configured in local backend")
    raise ValueError(f"Unknown LLM_MODE: {settings.llm_mode}")
