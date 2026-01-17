from app.llm.local_client import generate


def test_local_generate_smoke():
    text = generate("Hello", max_tokens=5)
    assert isinstance(text, str)
    assert len(text) > 0
