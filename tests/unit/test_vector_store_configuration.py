"""Unit tests for vector store fail-fast configuration."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.exceptions import ConfigurationError
from src.models.data_models import VectorStoreConfig
from src.services.vector_store_service import VectorStoreService, get_vector_store_service


class TestVectorStoreFailFast:
    """Test cases for vector store fail-fast initialization."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset singleton instance
        import src.services.vector_store_service
        src.services.vector_store_service._vector_store_service = None

    @patch('src.services.vector_store_service.chromadb')
    def test_vector_store_service_init_success(self, mock_chromadb):
        """Test successful VectorStoreService initialization."""
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Test service initialization
        service = VectorStoreService()
        assert service is not None
        
        # Verify ChromaDB client was created
        mock_chromadb.PersistentClient.assert_called_once()

    @patch('src.services.vector_store_service.chromadb')
    def test_vector_store_service_singleton(self, mock_chromadb):
        """Test VectorStoreService singleton pattern."""
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Get service instances
        service1 = get_vector_store_service()
        service2 = get_vector_store_service()
        
        # Verify they are the same instance
        assert service1 is service2

    @patch('src.services.vector_store_service.chromadb')
    def test_vector_store_service_add_item(self, mock_chromadb):
        """Test VectorStoreService add_item functionality."""
        # Mock ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Test add_item
        service = VectorStoreService()
        result = service.add_item("test_item", "test content", {"key": "value"})
        
        # Verify collection operations
        mock_client.get_or_create_collection.assert_called()
        mock_collection.add.assert_called_once()
        assert result is not None

    @patch('src.services.vector_store_service.chromadb')
    def test_vector_store_service_search(self, mock_chromadb):
        """Test VectorStoreService search functionality."""
        # Mock ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [['test document']],
            'metadatas': [[{'key': 'value'}]],
            'ids': [['test_id']],
            'distances': [[0.1]]
        }
        
        # Test search
        service = VectorStoreService()
        results = service.search("test query", k=5)
        
        # Verify search was called
        mock_collection.query.assert_called_once()
        assert len(results) == 1
        assert results[0]['content'] == 'test document'

    def teardown_method(self):
        """Clean up after each test."""
        # Reset singleton instance
        import src.services.vector_store_service
        src.services.vector_store_service._vector_store_service = None