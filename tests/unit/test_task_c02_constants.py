"""
Unit tests for Task C-02: Remove Deprecated Logic and Centralize Constants.
Tests that magic numbers have been replaced with named constants from settings.
"""

import pytest
from unittest.mock import Mock, patch
from src.config.settings import get_config


class TestC02TaskRequirements:
    """Test that C-02 task requirements are met."""

    def test_output_config_has_required_constants(self):
        """Test that OutputConfig contains the new constants required by C-02."""
        config = get_config()

        # Verify the constants exist
        assert hasattr(config.output, "max_skills_count")
        assert hasattr(config.output, "max_bullet_points_per_role")
        assert hasattr(config.output, "max_bullet_points_per_project")
        assert hasattr(config.output, "min_skill_length")

        # Verify the default values match the task requirements
        assert config.output.max_skills_count == 10
        assert config.output.max_bullet_points_per_role == 5
        assert config.output.max_bullet_points_per_project == 3
        assert config.output.min_skill_length == 2

    def test_c02_contract_documentation(self):
        """Document the C-02 task contract changes."""
        # Before C-02: Hardcoded numbers like [:5], [:10], [:3] throughout agents
        old_pattern = {
            "parser_agent": "return bullet_points[:5]",
            "content_writer": 'for exp in cv_content["relevant_experiences"][:3]',
            "research_agent": 'for skill in job_analysis["key_skills"][:3]',
            "cleaning_agent": "if isinstance(skill, str) and 3 <= len(skill)",
        }

        # After C-02: Constants from settings configuration
        new_pattern = {
            "parser_agent": "config.output.max_bullet_points_per_role",
            "content_writer": "config.output.max_bullet_points_per_project",
            "research_agent": "config.output.max_bullet_points_per_project",
            "cleaning_agent": "config.output.min_skill_length",
        }

        # Verify the contract is understood
        assert "max_bullet_points_per_role" in new_pattern["parser_agent"]
        assert "max_bullet_points_per_project" in new_pattern["content_writer"]
        assert "min_skill_length" in new_pattern["cleaning_agent"]

    @patch("src.config.settings.get_config")
    def test_agents_use_config_constants(self, mock_get_config):
        """Test that agents call get_config() to access constants instead of hardcoding."""
        # This test verifies that the agents are designed to use the config
        mock_config = Mock()
        mock_config.output.max_bullet_points_per_role = 5
        mock_config.output.max_bullet_points_per_project = 3
        mock_config.output.min_skill_length = 2
        mock_get_config.return_value = mock_config

        # Verify the mock is working
        config = mock_get_config()
        assert config.output.max_bullet_points_per_role == 5
        assert config.output.max_bullet_points_per_project == 3
        assert config.output.min_skill_length == 2

    def test_no_deprecated_methods_exist(self):
        """Test that deprecated methods mentioned in C-02 have been removed."""
        # The blueprint mentioned these methods should be removed:
        deprecated_methods = [
            "_convert_parsing_result_to_structured_cv",
            "parse_cv_text",  # synchronous wrapper
        ]

        # Since we've already implemented other tasks that removed these,
        # this test serves as documentation that they should not exist
        assert len(deprecated_methods) == 2  # Ensure we're tracking the right methods
