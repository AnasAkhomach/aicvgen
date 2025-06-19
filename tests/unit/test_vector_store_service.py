import pytest
from unittest.mock import patch, MagicMock
from src.services.vector_store_service import VectorStoreService, ConfigurationError


@patch("src.services.vector_store_service.chromadb.PersistentClient")
def test_vector_store_service_success(mock_client):
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.heartbeat.return_value = True
    service = VectorStoreService(
        settings=type("Settings", (), {"vector_db": type("VectorDB", (), {"persist_directory": "dummy_path"})()})()
    )
    assert service.get_client() == mock_instance


def test_vector_store_service_fail():
    with patch(
        "src.services.vector_store_service.chromadb.PersistentClient",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(ConfigurationError):
            VectorStoreService(
                settings=type("Settings", (), {"vector_db": type("VectorDB", (), {"persist_directory": "bad_path"})()})()
            )
