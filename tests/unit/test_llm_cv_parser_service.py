"""
Unit tests for LLMCVParserService (LLM orchestration decoupling)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.services.llm_cv_parser_service import LLMCVParserService
from src.models.data_models import JobDescriptionData, CVParsingResult


@pytest.mark.asyncio
async def test_parse_cv_with_llm_success(monkeypatch):
    # Mock LLM service
    mock_llm_service = MagicMock()
    mock_llm_service.generate = AsyncMock(
        return_value='{"personal_info": {"name": "John Doe", "email": "john@example.com", "phone": "1234567890", "linkedin": "", "github": "", "location": ""}, "sections": []}'
    )

    # Mock settings
    class DummySettings:
        def get_prompt_path_by_key(self, key):
            return "tests/unit/dummy_cv_prompt.txt"

    # Write dummy prompt file
    with open("tests/unit/dummy_cv_prompt.txt", "w", encoding="utf-8") as f:
        f.write("{{raw_cv_text}}")
    service = LLMCVParserService(mock_llm_service, DummySettings())
    result = await service.parse_cv_with_llm("CV TEXT")
    assert isinstance(result, CVParsingResult)
    assert result.personal_info.name == "John Doe"


@pytest.mark.asyncio
async def test_parse_job_description_with_llm_success(monkeypatch):
    mock_llm_service = MagicMock()
    mock_llm_service.generate = AsyncMock(
        return_value='{"skills": ["Python"], "experience_level": "Senior", "responsibilities": [], "industry_terms": [], "company_values": [], "raw_text": "JOB DESC"}'
    )

    class DummySettings:
        def get_prompt_path_by_key(self, key):
            return "tests/unit/dummy_job_prompt.txt"

    with open("tests/unit/dummy_job_prompt.txt", "w", encoding="utf-8") as f:
        f.write("{{raw_job_description}}")
    service = LLMCVParserService(mock_llm_service, DummySettings())
    result = await service.parse_job_description_with_llm("JOB DESC")
    assert isinstance(result, JobDescriptionData)
    assert result.skills == ["Python"]
    assert result.experience_level == "Senior"


@pytest.mark.asyncio
async def test_parse_cv_with_llm_llm_error():
    mock_llm_service = MagicMock()
    mock_llm_service.generate = AsyncMock(side_effect=Exception("LLM error"))

    class DummySettings:
        def get_prompt_path_by_key(self, key):
            return "tests/unit/dummy_cv_prompt.txt"

    with open("tests/unit/dummy_cv_prompt.txt", "w", encoding="utf-8") as f:
        f.write("{{raw_cv_text}}")
    service = LLMCVParserService(mock_llm_service, DummySettings())
    with pytest.raises(Exception):
        await service.parse_cv_with_llm("CV TEXT")
