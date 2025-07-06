"""Tests for vector store service contract breach fixes (CB-013)."""

import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.services.vector_store_service import VectorStoreService, MockVectorStoreService
from src.error_handling.exceptions import VectorStoreError
from src.config.settings import VectorDBConfig


class TestVectorStoreServiceContractFixes:
    """Test contract breach fixes for vector store service."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db_config = VectorDBConfig(
            persist_directory=self.temp_dir,
            collection_name="test_collection"
        )
        # Use vector_db_config directly as VectorStoreService now expects vector_config
        self.config = self.vector_db_config

    def teardown_method(self):
        """Clean up test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_add_item_uses_upsert_for_consistency(self):
        """Test that add_item uses upsert operation for data consistency."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            service = VectorStoreService(self.config)
            
            # Test add_item
            test_item = {"test": "data"}
            test_content = "test content"
            test_metadata = {"type": "test"}
            
            service.add_item(test_item, test_content, test_metadata)
            
            # Verify upsert was called
            mock_collection.upsert.assert_called_once()
            call_args = mock_collection.upsert.call_args
            
            assert "documents" in call_args.kwargs
            assert "metadatas" in call_args.kwargs
            assert "ids" in call_args.kwargs
            assert call_args.kwargs["documents"] == [test_content]

    def test_add_item_verifies_persistence(self):
        """Test that add_item verifies data was persisted correctly."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Mock successful verification
            mock_collection.get.return_value = {"ids": ["test_id"]}
            
            service = VectorStoreService(self.config)
            
            test_item = {"test": "data"}
            test_content = "test content"
            
            result_id = service.add_item(test_item, test_content)
            
            # Verify both upsert and get were called
            mock_collection.upsert.assert_called_once()
            mock_collection.get.assert_called_once()
            assert result_id is not None

    def test_persistence_integrity_verification(self):
        """Test the persistence integrity verification method."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Mock successful integrity check
            mock_collection.get.return_value = {"ids": ["integrity_test_" + str(hash("test_content"))]}
            
            service = VectorStoreService(self.config)
            
            result = service.verify_persistence_integrity()
            
            assert result is True
            # Verify test operations were performed
            mock_collection.upsert.assert_called_once()
            mock_collection.get.assert_called_once()
            mock_collection.delete.assert_called_once()

    def test_persistence_integrity_verification_failure(self):
        """Test persistence integrity verification handles failures."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Mock failed integrity check
            mock_collection.get.return_value = {"ids": []}
            
            service = VectorStoreService(self.config)
            
            result = service.verify_persistence_integrity()
            
            assert result is False

    def test_batch_add_items_consistency(self):
        """Test batch add items for better consistency."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Mock successful batch verification
            mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
            
            service = VectorStoreService(self.config)
            
            items_data = [
                ({"item": 1}, "content1", {"meta": "1"}),
                ({"item": 2}, "content2", {"meta": "2"}),
                ({"item": 3}, "content3", {"meta": "3"})
            ]
            
            result_ids = service.batch_add_items(items_data)
            
            # Verify batch upsert was called
            mock_collection.upsert.assert_called_once()
            call_args = mock_collection.upsert.call_args
            
            assert len(call_args.kwargs["documents"]) == 3
            assert len(call_args.kwargs["metadatas"]) == 3
            assert len(call_args.kwargs["ids"]) == 3
            assert len(result_ids) == 3

    def test_batch_add_items_empty_list(self):
        """Test batch add items handles empty input."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            service = VectorStoreService(self.config)
            
            result_ids = service.batch_add_items([])
            
            assert result_ids == []
            mock_collection.upsert.assert_not_called()

    def test_error_recovery_for_persistence_failures(self):
        """Test error recovery when persistence operations fail."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Mock ChromaDB error during upsert
            import chromadb.errors
            mock_collection.upsert.side_effect = chromadb.errors.ChromaError("Persistence failed")
            
            service = VectorStoreService(self.config)
            
            test_item = {"test": "data"}
            test_content = "test content"
            
            with pytest.raises(VectorStoreError):
                service.add_item(test_item, test_content)


class TestMockVectorStoreServiceContract:
    """Test that MockVectorStoreService maintains the same interface."""

    def test_mock_service_interface_consistency(self):
        """Test that mock service has the same interface as real service."""
        mock_service = MockVectorStoreService()
        
        # Test all methods exist and work
        assert hasattr(mock_service, 'add_item')
        assert hasattr(mock_service, 'search')
        assert hasattr(mock_service, 'get_client')
        assert hasattr(mock_service, 'shutdown')
        assert hasattr(mock_service, 'verify_persistence_integrity')
        assert hasattr(mock_service, 'batch_add_items')
        
        # Test method calls work
        result_id = mock_service.add_item({"test": "item"}, "content")
        assert result_id == "mock_id"
        
        search_results = mock_service.search("query")
        assert search_results == []
        
        integrity_result = mock_service.verify_persistence_integrity()
        assert integrity_result is True
        
        batch_results = mock_service.batch_add_items([("item1", "content1", None), ("item2", "content2", None)])
        assert len(batch_results) == 2
        assert all(id.startswith("mock_id_") for id in batch_results)

    def test_mock_service_batch_add_items(self):
        """Test mock service batch add items method."""
        mock_service = MockVectorStoreService()
        
        items_data = [
            ({"item": 1}, "content1", {"meta": "1"}),
            ({"item": 2}, "content2", {"meta": "2"})
        ]
        
        result_ids = mock_service.batch_add_items(items_data)
        
        assert len(result_ids) == 2
        assert result_ids == ["mock_id_0", "mock_id_1"]