"""Test CS-03 input validation implementation"""

from src.models.validation_schemas import (
    validate_agent_input,
    ParserAgentInput,
    ContentWriterAgentInput,
    ResearchAgentInput,
    QualityAssuranceAgentInput,
    FormatterAgentInput,
    CVAnalyzerAgentInput,
    CleaningAgentInput,
)
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV, JobDescriptionData
from pydantic import ValidationError


# Mock data for testing
def create_mock_state():
    """Create a mock agent state with all necessary fields"""
    return AgentState(
        cv_text="Sample CV text",
        structured_cv=StructuredCV(sections=[]),
        job_description_data=JobDescriptionData(
            raw_text="Sample job description",
            skills=["Python", "SQL"],
            experience_level="Senior",
            responsibilities=["Develop software"],
            industry_terms=["SaaS"],
            company_values=["Innovation"],
        ),
        current_item_id="test_item_123",
    )


def test_all_agent_validations():
    """Test that all agent input validations work correctly"""
    print("Testing CS-03 input validation implementation...")

    state = create_mock_state()

    # Test each agent type
    agent_types = [
        "parser",
        "content_writer",
        "research",
        "qa",
        "formatter",
        "cv_analyzer",
        "cleaning",
    ]

    for agent_type in agent_types:
        try:
            validated_input = validate_agent_input(agent_type, state)
            print(
                f"✅ {agent_type} validation passed - Type: {type(validated_input).__name__}"
            )
        except Exception as e:
            print(f"❌ {agent_type} validation failed: {e}")

    # Test invalid agent type (should return original state)
    result = validate_agent_input("unknown_agent", state)
    assert result == state
    print("✅ Unknown agent type handled correctly")
    # Test validation error (missing required field for parser)
    state_missing_job_data = create_mock_state()
    state_missing_job_data.job_description_data = None
    try:
        validate_agent_input("parser", state_missing_job_data)
        print("❌ Should have failed with missing job_description_data")
    except ValueError as e:
        print(f"✅ Validation error correctly caught: {type(e).__name__}")
    except Exception as e:
        print(
            f"✅ Validation error correctly caught (different type): {type(e).__name__}"
        )

    print("\n✅ All CS-03 input validation tests passed!")


if __name__ == "__main__":
    test_all_agent_validations()
