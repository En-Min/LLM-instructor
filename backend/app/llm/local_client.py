from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_MODEL_NAME = "sshleifer/tiny-gpt2"
_TOKENIZER = None
_MODEL = None


def _load_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME, cache_dir=str(cache_dir))
        _MODEL = AutoModelForCausalLM.from_pretrained(_MODEL_NAME, cache_dir=str(cache_dir))
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def generate(prompt: str, max_tokens: int = 64, seed: Optional[int] = None) -> str:
    tokenizer, model = _load_model()
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
