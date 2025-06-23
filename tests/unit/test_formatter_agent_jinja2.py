import pytest
from src.agents.formatter_agent import FormatterAgent
from src.models.data_models import StructuredCV, Section, Item, MetadataModel
from unittest.mock import Mock


def make_structured_cv():
    metadata = MetadataModel(
        item_id="meta-1",
        company=None,
        position=None,
        location=None,
        start_date=None,
        end_date=None,
        status=None,
        processing_time_seconds=None,
        tokens_used=None,
        extra={},
    )
    sections = [
        Section(
            name="Experience",
            items=[
                Item(content="Did X"),
                Item(content="Did Y"),
            ],
            subsections=[],
        ),
        Section(
            name="Skills",
            items=[
                Item(content="Python"),
                Item(content="ML"),
            ],
            subsections=[],
        ),
    ]
    return StructuredCV(
        sections=sections,
        metadata=metadata,
    )


@pytest.mark.parametrize("cv", [make_structured_cv()])
def test_format_with_jinja2(cv):
    agent = FormatterAgent(
        llm_service=Mock(),
        error_recovery_service=Mock(),
        progress_tracker=Mock(),
    )
    output = agent._format_with_jinja2(cv)
    assert "Experience" in output
    assert "Skills" in output
    assert "Did X" in output
    assert "Did Y" in output
    assert "Python" in output
    assert "ML" in output
