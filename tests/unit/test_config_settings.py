"""Unit tests for centralized configuration settings."""

import pytest
import os
from unittest.mock import patch, mock_open
from pathlib import Path

from src.config.settings import AppConfig, LLMSettings, PromptSettings


class TestLLMSettings:
    """Test cases for LLMSettings Pydantic model."""
    
    def test_llm_settings_defaults(self):
        """Test LLMSettings default values."""
        settings = LLMSettings()
        
        assert settings.default_model == "gemini-2.0-flash"
        assert settings.default_temperature == 0.7
        assert settings.max_tokens == 4096
    
    def test_llm_settings_custom_values(self):
        """Test LLMSettings with custom values."""
        settings = LLMSettings(
            default_model="gemini-1.5-pro",
            default_temperature=0.5,
            max_tokens=2000
        )
        
        assert settings.default_model == "gemini-1.5-pro"
        assert settings.default_temperature == 0.5
        assert settings.max_tokens == 2000


class TestPromptSettings:
    """Test cases for PromptSettings Pydantic model."""
    
    def test_prompt_settings_defaults(self):
        """Test that PromptSettings has correct default prompt filenames."""
        settings = PromptSettings()
        
        assert settings.job_description_parser == "job_description_parsing_prompt.md"
        assert settings.resume_role_writer == "resume_role_prompt.md"
        assert settings.project_writer == "side_project_prompt.md"
        assert settings.key_qualifications_writer == "key_qualifications_prompt.md"
        assert settings.executive_summary_writer == "executive_summary_prompt.md"
        assert settings.cv_analysis == "cv_analysis_prompt.md"
        assert settings.cv_assessment == "cv_assessment_prompt.md"
        assert settings.clean_big_6 == "clean_big_6_prompt.md"
        assert settings.clean_json_output == "clean_json_output_prompt.md"
        assert settings.job_research_analysis == "job_research_analysis_prompt.md"
    
    def test_prompt_settings_custom_values(self):
        """Test that PromptSettings accepts custom prompt filenames."""
        settings = PromptSettings(
            job_description_parser="custom_job_parser.md",
            resume_role_writer="custom_resume_writer.md"
        )
        
        assert settings.job_description_parser == "custom_job_parser.md"
        assert settings.resume_role_writer == "custom_resume_writer.md"
        # Other values should remain default
        assert settings.project_writer == "side_project_prompt.md"


class TestAppConfigIntegration:
    """Test cases for AppConfig integration with new settings."""
    
    def test_app_config_includes_new_settings(self):
        """Test that AppConfig includes the new LLMSettings and PromptSettings."""
        config = AppConfig()
        
        assert hasattr(config, 'llm_settings')
        assert hasattr(config, 'prompts')
        assert isinstance(config.llm_settings, LLMSettings)
        assert isinstance(config.prompts, PromptSettings)
    
    def test_get_prompt_path_by_key_success(self):
        """Test successful prompt path retrieval by key."""
        config = AppConfig()
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            path = config.get_prompt_path_by_key("job_description_parser")
            
            expected_filename = config.prompts.job_description_parser
            expected_path = os.path.join(str(config.prompts_directory), expected_filename)
            assert path == expected_path
    
    def test_get_prompt_path_by_key_invalid_key(self):
        """Test error handling for invalid prompt key."""
        config = AppConfig()
        
        with pytest.raises(ValueError, match="Prompt key 'invalid_key' not found in settings"):
            config.get_prompt_path_by_key("invalid_key")
    
    def test_get_prompt_path_by_key_file_not_exists(self):
        """Test error handling when prompt file doesn't exist."""
        config = AppConfig()
        
        # Mock file doesn't exist
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Prompt file does not exist at path"):
                config.get_prompt_path_by_key("job_description_parser")
    
    def test_get_prompt_path_by_key_all_keys(self):
        """Test that all prompt keys can be retrieved successfully."""
        config = AppConfig()
        prompt_keys = [
            "job_description_parser",
            "resume_role_writer", 
            "project_writer",
            "key_qualifications_writer",
            "executive_summary_writer",
            "cv_analysis",
            "cv_assessment",
            "clean_big_6",
            "clean_json_output",
            "job_research_analysis"
        ]
        
        with patch('os.path.exists', return_value=True):
            for key in prompt_keys:
                path = config.get_prompt_path_by_key(key)
                assert path is not None
                assert isinstance(path, str)
                assert path.endswith('.md')


class TestBackwardsCompatibility:
    """Test that existing functionality still works after refactoring."""
    
    def test_existing_get_prompt_path_still_works(self):
        """Test that the original get_prompt_path method still works."""
        config = AppConfig()
        
        path = config.get_prompt_path("test_prompt")
        expected = config.prompts_directory / "test_prompt.md"
        
        assert path == expected
    
    def test_llm_config_still_exists(self):
        """Test that the original LLMConfig dataclass still exists."""
        config = AppConfig()
        
        assert hasattr(config, 'llm')
        assert config.llm is not None
        # Should have both old and new configurations
        assert hasattr(config, 'llm_settings')