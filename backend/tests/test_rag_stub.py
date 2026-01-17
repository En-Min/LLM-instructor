from app.rag.retriever import RAGRetriever


def test_rag_stub_returns_list():
    retriever = RAGRetriever()
    results = retriever.search("attention", top_k=2)
    assert isinstance(results, list)
