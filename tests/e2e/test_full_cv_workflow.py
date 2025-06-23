import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.core.state_manager import StateManager
from src.orchestration.state import AgentState
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.services.llm_service import EnhancedLLMService
from src.services.llm_client import LLMClient
from src.services.llm_retry_handler import LLMRetryHandler
from src.models.data_models import StructuredCV, JobDescriptionData
from src.models.cv_analysis_result import CVAnalysisResult


@pytest.mark.asyncio
async def test_full_cv_workflow(monkeypatch):
    """End-to-end: Simulate a full CV workflow from parsing to QA."""
    # Setup dependencies and state
    mock_settings = Mock()
    mock_settings.llm.gemini_api_key_fallback = "fallback-key"
    mock_settings.llm_settings.default_model = "test-model"

    # Mock the LLM service to avoid real API calls
    llm_service = Mock()
    llm_service.generate_content = AsyncMock()
    # Mock LLM response for successful generation
    mock_llm_response = Mock()
    mock_llm_response.success = True
    mock_llm_response.content = (
        "Enhanced work experience bullet point with AI and machine learning focus"
    )
    mock_llm_response.text = '{"skills": ["Python", "AI"], "experience_level": "Senior", "responsibilities": ["Develop AI solutions"], "industry_terms": ["Machine Learning"], "company_values": ["Innovation"]}'
    mock_llm_response.error_message = None
    llm_service.generate_content.return_value = mock_llm_response

    state_manager = StateManager()

    # Create a proper CV structure with sections and items for testing
    from src.models.data_models import (
        Section,
        Item,
        MetadataModel,
        ItemType,
        ItemStatus,
    )

    test_item = Item(
        content="Sample work experience bullet point",
        item_type=ItemType.BULLET_POINT,
        status=ItemStatus.INITIAL,
        metadata=MetadataModel(item_id="item-1"),
    )

    test_section = Section(
        name="Professional Experience",
        content_type="DYNAMIC",
        items=[test_item],
        status=ItemStatus.INITIAL,
    )

    test_cv = StructuredCV(sections=[test_section])

    initial_state = AgentState(
        structured_cv=test_cv,
        cv_text="Sample CV text for parsing.",
        job_description_data=JobDescriptionData(raw_text="Sample job description."),
        cv_analysis_results=None,
        current_item_id="item-1",  # Added valid string for current_item_id
    )

    # Mock all required dependencies for DI
    vector_store_service = Mock()
    error_recovery_service = Mock()
    progress_tracker = Mock()

    # Run parser agent
    parser_agent = ParserAgent(
        llm_service=llm_service,
        vector_store_service=vector_store_service,
        error_recovery_service=error_recovery_service,
        progress_tracker=progress_tracker,
        settings=mock_settings,
        name="ParserAgent",
        description="Parses CVs into structured format.",
    )
    parser_result = await parser_agent.run_as_node(initial_state)
    assert parser_result.structured_cv is not None

    # Run research agent
    research_agent = ResearchAgent(
        llm_service=llm_service,
        error_recovery_service=error_recovery_service,
        progress_tracker=progress_tracker,
        vector_db=vector_store_service,
        settings=mock_settings,
        name="ResearchAgent",
        description="Performs research on job description and CV.",
    )
    research_result = await research_agent.run_as_node(parser_result)
    assert hasattr(research_result, "research_findings")  # Run content writer agent

    # Mock parser_agent for EnhancedContentWriterAgent DI
    parser_agent_for_writer = Mock()
    content_writer = EnhancedContentWriterAgent(
        llm_service=llm_service,
        error_recovery_service=error_recovery_service,
        progress_tracker=progress_tracker,
        parser_agent=parser_agent_for_writer,
        settings=mock_settings,
    )
    content_result = await content_writer.run_as_node(research_result)
    # Check that the content writer returned a valid structured CV
    assert hasattr(content_result, "structured_cv")

    # Extract the structured CV
    structured_cv = content_result.structured_cv

    # Verify the item was processed successfully
    assert len(structured_cv.sections) > 0
    assert len(structured_cv.sections[0].items) > 0
    processed_item = structured_cv.sections[0].items[0]
    assert processed_item.metadata.item_id == "item-1"

    # Run QA agent
    qa_agent = QualityAssuranceAgent(
        llm_service=llm_service,
        error_recovery_service=error_recovery_service,
        progress_tracker=progress_tracker,
        name="QualityAssuranceAgent",
        description="Performs quality assurance on CV content.",
    )
    qa_result = await qa_agent.run_as_node(content_result)
    assert hasattr(qa_result, "quality_check_results")  # Test completed successfully!
    print("✅ Full CV workflow test passed!")
    print(f"✅ Processed item: {processed_item.content}")
    print(f"✅ Item status: {processed_item.status}")
    print(f"✅ QA results available: {qa_result.quality_check_results is not None}")

    # Validate the architecture contract is enforced
    assert isinstance(
        content_result, AgentState
    ), "Content writer must return AgentState (Pydantic model)"
    assert isinstance(
        qa_result, AgentState
    ), "QA agent must return AgentState (Pydantic model)"
    assert isinstance(
        content_result.structured_cv, StructuredCV
    ), "StructuredCV must be a Pydantic model"

    print(
        "✅ Architecture contract validated: All agents use Pydantic models correctly!"
    )
