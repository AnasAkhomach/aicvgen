"""Unit test for ParserAgent full conversion logic."""

import pytest
from unittest.mock import MagicMock
from src.agents.parser_agent import ParserAgent
from src.models.data_models import JobDescriptionData, StructuredCV


class DummyLLMService:
    async def parse_cv_with_llm(self, cv_text, session_id=None, trace_id=None):
        class DummyParsingResult:
            personal_info = type(
                "PersonalInfo",
                (),
                dict(
                    name="John Doe",
                    email="john@example.com",
                    phone="123",
                    linkedin="",
                    github="",
                    location="Earth",
                ),
            )()
            sections = [
                type(
                    "Section",
                    (),
                    dict(
                        name="Professional Experience",
                        items=["Did stuff"],
                        subsections=[],
                    ),
                )(),
                type(
                    "Section", (), dict(name="Education", items=["BSc"], subsections=[])
                )(),
            ]

        return DummyParsingResult()

    async def parse_job_description_with_llm(
        self, raw_text, session_id=None, trace_id=None
    ):
        return JobDescriptionData(
            raw_text=raw_text, title="Engineer", company_name="ACME"
        )

    async def generate(self, prompt, session_id=None, trace_id=None):
        return '{"personal_info": {"name": "John Doe", "email": "john@example.com", "phone": "123", "linkedin": "", "github": "", "location": "Earth"}, "sections": [{"name": "Professional Experience", "items": ["Did stuff"], "subsections": []}, {"name": "Education", "items": ["BSc"], "subsections": []}]}'


class DummyVectorStoreService:
    pass


class DummyProgressTracker:
    pass


class DummySettings:
    pass


class DummyTemplateManager:
    def get_template(self, *args, **kwargs):
        return "DUMMY TEMPLATE"

    def format_template(self, template, context):
        return template  # Just return the template string for test purposes


def make_agent():
    return ParserAgent(
        llm_service=DummyLLMService(),
        vector_store_service=DummyVectorStoreService(),
        progress_tracker=DummyProgressTracker(),
        settings=DummySettings(),
        template_manager=DummyTemplateManager(),
    )


def test_parser_agent_full_conversion():
    import asyncio

    agent = make_agent()
    cv_text = "Sample CV text"
    job_data = JobDescriptionData(raw_text="JD", title="Engineer", company_name="ACME")
    result = asyncio.run(agent.parse_cv_with_llm(cv_text, job_data))
    assert isinstance(result, StructuredCV)
    assert any(s.name == "Professional Experience" for s in result.sections)
    assert any(s.name == "Education" for s in result.sections)
