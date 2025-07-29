"""Test to verify the directory restructuring was completed successfully."""
import os
from pathlib import Path

import pytest


class TestDirectoryRestructuring:
    """Test class to verify the foundational directory restructuring."""

    def test_new_directories_exist(self):
        """Test that all required new directories were created."""
        base_path = Path("src/core")

        required_dirs = [
            base_path / "containers",
            base_path / "facades",
            base_path / "managers",
            base_path / "utils",
        ]

        for directory in required_dirs:
            assert directory.exists(), f"Directory {directory} should exist"
            assert directory.is_dir(), f"{directory} should be a directory"

    def test_files_moved_correctly(self):
        """Test that files were moved to their new locations."""
        moved_files = [
            "src/core/containers/main_container.py",
            "src/core/managers/workflow_manager.py",
            "src/core/managers/session_manager.py",
        ]

        for file_path in moved_files:
            path = Path(file_path)
            assert path.exists(), f"File {file_path} should exist in new location"
            assert path.is_file(), f"{file_path} should be a file"

    def test_old_files_removed(self):
        """Test that files were removed from old locations."""
        old_files = [
            "src/core/container.py",
            "src/core/workflow_manager.py",
            "src/services/session_manager.py",
        ]

        for file_path in old_files:
            path = Path(file_path)
            assert (
                not path.exists()
            ), f"File {file_path} should not exist in old location"

    def test_utils_directory_moved(self):
        """Test that utils files were moved to core/utils."""
        utils_files = [
            "__init__.py",
            "cv_data_factory.py",
            "decorators.py",
            "import_fallbacks.py",
            "json_utils.py",
            "latex_utils.py",
            "node_validation.py",
            "performance.py",
            "prompt_utils.py",
            "retry_predicates.py",
            "security_utils.py",
            "state_utils.py",
            "streamlit_utils.py",
        ]

        for file_name in utils_files:
            new_path = Path(f"src/core/utils/{file_name}")
            old_path = Path(f"src/utils/{file_name}")

            assert (
                new_path.exists()
            ), f"File {file_name} should exist in src/core/utils/"
            assert (
                not old_path.exists()
            ), f"File {file_name} should not exist in old src/utils/"

    def test_old_utils_directory_removed(self):
        """Test that the old utils directory was completely removed."""
        old_utils_path = Path("src/utils")
        assert (
            not old_utils_path.exists()
        ), "Old src/utils directory should be completely removed"
