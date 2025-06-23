import pytest
from src.services.vector_store_service import VectorStoreService
from src.models.vector_store_and_session_models import VectorStoreSearchResult


class DummyItem:
    def __init__(self, id, label):
        self.id = id
        self.label = label


def test_vector_store_search_returns_pydantic_models(monkeypatch):
    from src.config.settings import AppConfig

    service = VectorStoreService(settings=AppConfig())
    # Patch collection.query to return a fake result
    monkeypatch.setattr(
        service,
        "collection",
        type(
            "FakeCollection",
            (),
            {
                "query": lambda self, query_texts, n_results: {
                    "documents": [["doc1", "doc2"]],
                    "metadatas": [[{"foo": "bar"}, {"foo": "baz"}]],
                    "ids": [["id1", "id2"]],
                    "distances": [[0.1, 0.2]],
                }
            },
        )(),
    )
    results = service.search("test", k=2)
    assert all(isinstance(r, VectorStoreSearchResult) for r in results)
    assert results[0].content == "doc1"
    assert results[0].metadata == {"foo": "bar"}
    assert results[0].id == "id1"
    assert results[0].distance == 0.1
