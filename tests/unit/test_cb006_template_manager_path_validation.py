"""Tests for CB-006: Template manager path validation fix."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.container import validate_prompts_directory


class TestCB006TemplateManagerPathValidation:
    """Test suite for CB-006 template manager path validation."""

    def test_validate_existing_directory(self):
        """Test validation with an existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_prompts_directory(temp_dir)
            assert result == temp_dir

    def test_validate_nonexistent_directory_creates_it(self):
        """Test that nonexistent directory gets created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to ensure fallback paths don't exist
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                nonexistent_path = Path(temp_dir) / "new_prompts"
                result = validate_prompts_directory(str(nonexistent_path))
                
                assert result == str(nonexistent_path)
                assert nonexistent_path.exists()
                assert nonexistent_path.is_dir()
            finally:
                os.chdir(original_cwd)

    def test_validate_fallback_to_data_prompts(self):
        """Test fallback to data/prompts when configured path doesn't exist."""
        # Create a temporary data/prompts directory
        with tempfile.TemporaryDirectory() as temp_dir:
            data_prompts = Path(temp_dir) / "data" / "prompts"
            data_prompts.mkdir(parents=True)
            
            # Change to the temp directory so fallback paths work
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Try with a nonexistent path that can't be created
                with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
                    result = validate_prompts_directory("/invalid/path")
                    # The function returns the relative path "data/prompts" when using fallback
                    # On Windows, this will be "data\\prompts"
                    expected_path = str(Path("data/prompts"))
                    assert result == expected_path
            finally:
                os.chdir(original_cwd)

    def test_validate_permission_error_raises_runtime_error(self):
        """Test that permission errors raise RuntimeError."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            
            with pytest.raises(RuntimeError, match="Cannot access or create prompts directory"):
                validate_prompts_directory("/invalid/path")

    def test_validate_relative_path(self):
        """Test validation with relative path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a relative path within the temp directory
            rel_path = Path(temp_dir) / "relative_prompts"
            rel_path.mkdir()
            
            result = validate_prompts_directory(str(rel_path))
            assert result == str(rel_path)

    def test_validate_file_instead_of_directory_creates_directory(self):
        """Test behavior when path points to a file instead of directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file at the path
            file_path = Path(temp_dir) / "prompts_file"
            file_path.touch()
            
            # Change to temp directory for fallback to work
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Create data/prompts fallback directory
                fallback_dir = Path(temp_dir) / "data" / "prompts"
                fallback_dir.mkdir(parents=True)
                
                # Since the file exists but is not a directory, it should use fallback
                result = validate_prompts_directory(str(file_path))
                # On Windows, this will be "data\\prompts"
                expected_path = str(Path("data/prompts"))
                assert result == expected_path
            finally:
                os.chdir(original_cwd)

    @patch('src.core.container.logger')
    def test_logging_on_successful_validation(self, mock_logger):
        """Test that successful validation logs appropriate message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validate_prompts_directory(temp_dir)
            
            mock_logger.info.assert_called_once_with(
                "Prompts directory validated",
                extra={"directory": str(Path(temp_dir).resolve())}
            )

    @patch('src.core.container.logger')
    def test_logging_on_fallback_usage(self, mock_logger):
        """Test that fallback usage logs warning message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_prompts = Path(temp_dir) / "data" / "prompts"
            data_prompts.mkdir(parents=True)
            
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
                    validate_prompts_directory("/invalid/path")
                    
                    mock_logger.warning.assert_called_once_with(
                        "Using fallback prompts directory",
                        extra={
                            "configured_path": "/invalid/path",
                            "fallback_path": str(data_prompts.resolve())
                        }
                    )
            finally:
                os.chdir(original_cwd)

    @patch('src.core.container.logger')
    def test_logging_on_directory_creation(self, mock_logger):
        """Test that directory creation logs appropriate message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to ensure fallback paths don't exist
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                new_dir = Path(temp_dir) / "new_prompts"
                validate_prompts_directory(str(new_dir))
                
                mock_logger.info.assert_called_with(
                    "Created missing prompts directory",
                    extra={"directory": str(new_dir.resolve())}
                )
            finally:
                os.chdir(original_cwd)