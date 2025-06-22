import pytest
from pydantic import ValidationError
from src.models.validation_schemas import (
    validate_agent_input,
    ParserAgentInput,
    ContentWriterAgentInput,
    ResearchAgentInput,
    QualityAssuranceAgentInput,
)
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV, JobDescriptionData
from src.models.research_models import ResearchFindings


def test_parser_agent_input_validation():
    state = AgentState(
        cv_text="Sample CV text",
        job_description_data=JobDescriptionData(raw_text="Engineer JD raw text"),
        structured_cv=StructuredCV(),
    )
    result = validate_agent_input("parser", state)
    assert isinstance(result, ParserAgentInput)
    assert result.cv_text == "Sample CV text"


def test_content_writer_agent_input_validation():
    state = AgentState(
        structured_cv=StructuredCV(),
        research_findings=ResearchFindings(),
        current_item_id="item1",
    )
    result = validate_agent_input("content_writer", state)
    assert isinstance(result, ContentWriterAgentInput)
    assert result.current_item_id == "item1"


def test_research_agent_input_validation():
    state = AgentState(
        job_description_data=JobDescriptionData(raw_text="Engineer JD raw text"),
        structured_cv=StructuredCV(),
    )
    result = validate_agent_input("research", state)
    assert isinstance(result, ResearchAgentInput)


def test_quality_assurance_agent_input_validation():
    state = AgentState(
        structured_cv=StructuredCV(),
        current_item_id="item2",
    )
    result = validate_agent_input("qa", state)
    assert isinstance(result, QualityAssuranceAgentInput)
    assert result.current_item_id == "item2"


def test_invalid_input_raises():
    with pytest.raises(Exception):
        validate_agent_input("parser", AgentState(structured_cv=StructuredCV()))
