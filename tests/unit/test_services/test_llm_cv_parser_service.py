import pytest
from unittest.mock import Mock
from src.services.llm_cv_parser_service import LLMCVParserService
from src.templates.content_templates import (
    ContentTemplateManager,
    ContentTemplate,
    TemplateCategory,
)
from src.config.settings import PromptSettings, Settings
from src.services.llm_service import EnhancedLLMService
from src.models.data_models import ContentType


@pytest.fixture
def prompt_settings():
    """Fixture for PromptSettings."""
    return PromptSettings(directory="data/prompts", cv_parser="cv_parsing_prompt_v2.md")


@pytest.fixture
def template_manager(prompt_settings):
    """Fixture for ContentTemplateManager."""
    manager = ContentTemplateManager(prompt_directory=prompt_settings.directory)
    mock_template = ContentTemplate(
        name="cv_parsing_prompt_v2",
        category=TemplateCategory.PROMPT,
        content_type=ContentType.CV_PARSING,
        template="Test content",
        variables=[],
        description="A test prompt",
    )
    manager.templates = {"cv_parsing_prompt_v2": mock_template}
    return manager


@pytest.fixture
def mock_llm_service():
    """Fixture for a mocked EnhancedLLMService."""
    return Mock(spec=EnhancedLLMService)


@pytest.fixture
def mock_settings(prompt_settings):
    """Fixture for a mocked Settings object."""
    settings = Mock(spec=Settings)
    settings.prompts = prompt_settings
    return settings
