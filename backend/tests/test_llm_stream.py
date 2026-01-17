from app.llm.local_client import generate_stream


def test_generate_stream_yields_text():
    chunks = list(generate_stream("Hello", max_tokens=8))
    assert any(chunk.strip() for chunk in chunks)
