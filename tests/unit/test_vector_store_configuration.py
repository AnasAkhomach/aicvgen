"""Unit tests for vector store fail-fast configuration."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.exceptions import ConfigurationError
from src.models.data_models import VectorStoreConfig
from src.services.vector_db import VectorDB, get_enhanced_vector_db, get_vector_store_service


class TestVectorStoreFailFast:
    """Test cases for vector store fail-fast initialization."""

    def setup_method(self):
        """Reset global state before each test."""
        # Reset singleton instance
        import src.services.vector_db
        src.services.vector_db._enhanced_vector_db = None

    def test_vector_db_init_with_invalid_config_none(self):
        """Test VectorDB initialization fails with None config."""
        with pytest.raises(ConfigurationError, match="VectorStoreConfig is required"):
            VectorDB(config=None)

    def test_vector_db_init_with_invalid_dimension(self):
        """Test VectorDB initialization fails with invalid dimension."""
        config = VectorStoreConfig(
            dimension=0,  # Invalid dimension
            index_type="IndexFlatL2"
        )
        with pytest.raises(ConfigurationError, match="Invalid dimension: 0. Must be positive."):
            VectorDB(config=config)

    def test_vector_db_init_with_invalid_index_type(self):
        """Test VectorDB initialization fails with invalid index type."""
        config = VectorStoreConfig(
            dimension=768,
            index_type="InvalidIndexType"
        )
        with pytest.raises(ConfigurationError, match="Invalid index type: InvalidIndexType"):
            VectorDB(config=config)

    def test_vector_db_init_with_invalid_db_path_permissions(self):
        """Test VectorDB initialization fails with invalid database path permissions."""
        config = VectorStoreConfig(
            dimension=768,
            index_type="IndexFlatL2"
        )
        
        # Mock Path.mkdir to raise PermissionError
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(ConfigurationError, match="CRITICAL: Vector store initialization failed"):
                VectorDB(config=config, db_path="/invalid/path")

    def test_vector_db_init_with_corrupted_database(self):
        """Test VectorDB initialization fails with corrupted database."""
        config = VectorStoreConfig(
            dimension=768,
            index_type="IndexFlatL2"
        )
        
        # Mock _load_enhanced_database to raise an exception
        with patch.object(VectorDB, '_load_enhanced_database') as mock_load:
            mock_load.side_effect = Exception("Database corrupted")
            
            with pytest.raises(ConfigurationError, match="CRITICAL: Failed to load existing vector database"):
                VectorDB(config=config)

    def test_vector_db_init_success(self):
        """Test successful VectorDB initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(
                dimension=768,
                index_type="IndexFlatL2"
            )
            
            # Should not raise any exception
            db = VectorDB(config=config, db_path=temp_dir)
            assert db is not None
            assert db.config == config
            assert db.index is not None

    def test_get_enhanced_vector_db_singleton_behavior(self):
        """Test that get_enhanced_vector_db returns the same instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = VectorStoreConfig(
                dimension=768,
                index_type="IndexFlatL2"
            )
            
            # Mock the db_path to use temp directory
            with patch('src.services.vector_db.get_config') as mock_get_config:
                mock_config = MagicMock()
                mock_config.data_directory = temp_dir
                mock_get_config.return_value = mock_config
                
                db1 = get_enhanced_vector_db(config)
                db2 = get_enhanced_vector_db(config)
                
                assert db1 is db2  # Same instance

    def test_get_enhanced_vector_db_with_default_config(self):
        """Test get_enhanced_vector_db with default configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the db_path to use temp directory
            with patch('src.services.vector_db.get_config') as mock_get_config:
                mock_config = MagicMock()
                mock_config.data_directory = temp_dir
                mock_get_config.return_value = mock_config
                
                db = get_enhanced_vector_db()  # No config provided
                assert db is not None
                assert db.config.dimension == 768
                assert db.config.index_type == "IndexFlatL2"

    def test_get_enhanced_vector_db_propagates_configuration_error(self):
        """Test that get_enhanced_vector_db propagates ConfigurationError."""
        config = VectorStoreConfig(
            dimension=0,  # Invalid dimension
            index_type="IndexFlatL2"
        )
        
        with pytest.raises(ConfigurationError, match="Invalid dimension: 0. Must be positive."):
            get_enhanced_vector_db(config)

    def test_get_vector_store_service_alias(self):
        """Test that get_vector_store_service is an alias for get_enhanced_vector_db."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the db_path to use temp directory
            with patch('src.services.vector_db.get_config') as mock_get_config:
                mock_config = MagicMock()
                mock_config.data_directory = temp_dir
                mock_get_config.return_value = mock_config
                
                db1 = get_enhanced_vector_db()
                db2 = get_vector_store_service()
                
                assert db1 is db2  # Same instance

    def test_get_vector_store_service_propagates_configuration_error(self):
        """Test that get_vector_store_service propagates ConfigurationError."""
        # Reset singleton to force re-initialization
        import src.services.vector_db
        src.services.vector_db._enhanced_vector_db = None
        
        # Mock VectorDB to raise ConfigurationError
        with patch('src.services.vector_db.VectorDB') as mock_vector_db:
            mock_vector_db.side_effect = ConfigurationError("Test error")
            
            with pytest.raises(ConfigurationError, match="Test error"):
                get_vector_store_service()

    def teardown_method(self):
        """Clean up after each test."""
        # Reset singleton instance
        import src.services.vector_db
        src.services.vector_db._enhanced_vector_db = None