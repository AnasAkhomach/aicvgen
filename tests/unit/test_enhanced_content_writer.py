"""Unit tests for EnhancedContentWriterAgent (linter/runtime compliance and basic contract)."""

import pytest
from unittest.mock import MagicMock
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.models.data_models import AgentIO, ContentType


class DummyLLMService:
    async def generate_content(self, *args, **kwargs):
        class Response:
            success = True
            content = "Generated content"
            error_message = None
            metadata = {}
            processing_time = 3.0

        return Response()


class DummyProgressTracker:
    pass


class DummyParserAgent:
    def extract_skills(self, text):
        return ["Python", "AI"]


class DummySettings:
    def __init__(self, prompt_files):
        self.prompt_files = prompt_files

    def get_prompt_path_by_key(self, key):
        # Return the correct dummy prompt for the key
        return self.prompt_files.get(key, self.prompt_files["default"])


@pytest.fixture(scope="module")
def dummy_prompt_files(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")
    # Key qualifications prompt (only uses main_job_description_raw and my_talents)
    kq_prompt = tmpdir / "kq_prompt.txt"
    kq_prompt.write_text("Prompt: {main_job_description_raw} {my_talents}")
    # Default prompt (uses job_title, company_name, item_title, item_description)
    default_prompt = tmpdir / "default_prompt.txt"
    default_prompt.write_text(
        "Prompt: {job_title} {company_name} {item_title} {item_description}"
    )
    return {
        "key_qualifications_writer": str(kq_prompt),
        "default": str(default_prompt),
    }


@pytest.fixture
def agent(dummy_prompt_files, monkeypatch):
    settings = DummySettings(dummy_prompt_files)
    return EnhancedContentWriterAgent(
        llm_service=DummyLLMService(),
        progress_tracker=DummyProgressTracker(),
        parser_agent=DummyParserAgent(),
        settings=settings,
    )


def test_run_async_minimal(agent):
    input_data = {
        "structured_cv": {"sections": []},
        "current_item_id": "1",
        "job_description_data": {},
    }
    import asyncio

    result = asyncio.run(agent.run_async(input_data, context=None))
    assert hasattr(result, "success")
    assert hasattr(result, "output_data")


def test_generate_big_10_skills(agent):
    import asyncio

    result = asyncio.run(agent.generate_big_10_skills("Job description text", "Talent"))
    assert result["success"]
    assert "Python" in result["skills"]
    assert "AI" in result["skills"]
