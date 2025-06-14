"""End-to-End test for individual item processing workflow.

Tests the granular processing capabilities where users can
regenerate individual CV items while preserving others.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from uuid import uuid4

# Import application components
from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, Section, Subsection, Item, ItemStatus, ItemType,
    JobDescriptionData
)
from src.core.state_manager import StateManager
from src.services.llm import EnhancedLLMService as LLMService
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent


@pytest.mark.e2e
@pytest.mark.asyncio
class TestIndividualItemProcessing:
    """E2E tests for individual item processing workflow."""

    @pytest.fixture
    def sample_structured_cv_with_items(self):
        """Sample structured CV with multiple items for testing."""
        return StructuredCV(
            sections=[
                Section(
                    name="Professional Experience",
                    subsections=[
                        Subsection(
                            name="Senior Software Engineer @ TechCorp",
                            items=[
                                Item(
                                    id=uuid4(),
                                    content="Developed scalable web applications using Python and Django",
                                    status=ItemStatus.GENERATED,
                                    item_type=ItemType.BULLET_POINT,
                                    confidence_score=0.8,
                                    raw_llm_output="Original LLM response for item 1"
                                ),
                                Item(
                                    id=uuid4(),
                                    content="Implemented REST APIs for mobile applications",
                                    status=ItemStatus.GENERATED,
                                    item_type=ItemType.BULLET_POINT,
                                    confidence_score=0.75,
                                    raw_llm_output="Original LLM response for item 2"
                                ),
                                Item(
                                    id=uuid4(),
                                    content="Led cross-functional team of 5 developers",
                                    status=ItemStatus.GENERATED,
                                    item_type=ItemType.BULLET_POINT,
                                    confidence_score=0.9,
                                    raw_llm_output="Original LLM response for item 3"
                                )
                            ]
                        ),
                        Subsection(
                            name="Software Engineer @ StartupXYZ",
                            items=[
                                Item(
                                    id=uuid4(),
                                    content="Built frontend components using React and TypeScript",
                                    status=ItemStatus.GENERATED,
                                    item_type=ItemType.BULLET_POINT,
                                    confidence_score=0.85,
                                    raw_llm_output="Original LLM response for startup item 1"
                                ),
                                Item(
                                    id=uuid4(),
                                    content="Participated in agile development processes",
                                    status=ItemStatus.GENERATED,
                                    item_type=ItemType.BULLET_POINT,
                                    confidence_score=0.7,
                                    raw_llm_output="Original LLM response for startup item 2"
                                )
                            ]
                        )
                    ]
                ),
                Section(
                    name="Key Qualifications",
                    items=[
                        Item(
                            id=uuid4(),
                            content="5+ years of software engineering experience",
                            status=ItemStatus.GENERATED,
                            item_type=ItemType.KEY_QUALIFICATION,
                            confidence_score=0.95,
                            raw_llm_output="Key qualification LLM response 1"
                        ),
                        Item(
                            id=uuid4(),
                            content="Expert in Python, JavaScript, and cloud technologies",
                            status=ItemStatus.GENERATED,
                            item_type=ItemType.KEY_QUALIFICATION,
                            confidence_score=0.88,
                            raw_llm_output="Key qualification LLM response 2"
                        )
                    ]
                )
            ],
            big_10_skills=[
                "Python Programming",
                "Web Development",
                "API Design",
                "Team Leadership",
                "Agile Methodologies"
            ]
        )

    @pytest.fixture
    def sample_job_description_data(self):
        """Sample job description data for testing."""
        return JobDescriptionData(
            raw_text="Senior Software Engineer position at technology company focused on scalable web applications. Develop applications, lead teams, and design APIs using Python, Django, REST APIs, and Leadership skills in Software Engineering, Web Development, and Agile environment.",
            skills=["Python", "Django", "REST APIs", "Leadership"],
            responsibilities=["Develop applications", "Lead teams", "Design APIs"],
            industry_terms=["Software Engineering", "Web Development", "Agile"],
            company_values=["Scalability", "Innovation"]
        )

    @pytest.fixture
    def orchestrator_with_content_writer_mock(self):
        """Create orchestrator with mocked content writer for item processing."""
        # Mock content writer agent
        mock_content_writer = AsyncMock(spec=EnhancedContentWriterAgent)
        
        # Configure mock to return regenerated content
        async def mock_process_single_item(item_id, job_data, cv_data, section_key=None):
            return {
                "success": True,
                "item": Item(
                    id=item_id,
                    content="REGENERATED: Enhanced content with improved metrics and specific achievements",
                    status=ItemStatus.GENERATED,
                    item_type=ItemType.BULLET_POINT,
                    confidence_score=0.92,
                    raw_llm_output="Regenerated LLM response with enhanced details",
                    metadata={"regeneration_count": 1, "timestamp": "2024-01-01T12:00:00Z"}
                ),
                "metadata": {
                    "processing_time": 2.5,
                    "model_used": "gemini-pro",
                    "regeneration_reason": "user_requested"
                }
            }
        
        mock_content_writer.process_single_item = mock_process_single_item
        
        # Create orchestrator with mocked content writer
        from src.core.state_manager import StateManager
        state_manager = StateManager()
        orchestrator = EnhancedOrchestrator(state_manager=state_manager)
        orchestrator.content_writer_agent = mock_content_writer
        
        return orchestrator

    async def test_single_item_regeneration(self, orchestrator_with_content_writer_mock, sample_structured_cv_with_items, sample_job_description_data):
        """Test regenerating a single item while preserving others.
        
        Validates:
        - REQ-FUNC-GEN-3: Individual item processing
        - REQ-FUNC-GEN-4: User control over regeneration
        - REQ-FUNC-UI-2: Accept/Regenerate functionality
        """
        orchestrator = orchestrator_with_content_writer_mock
        session_id = str(uuid4())
        
        # Get the structured CV and select an item to regenerate
        structured_cv = sample_structured_cv_with_items
        target_item = structured_cv.sections[0].subsections[0].items[1]  # Second item in first role
        original_content = target_item.content
        original_raw_output = target_item.raw_llm_output
        
        # Store original content of other items for comparison
        other_items = [
            structured_cv.sections[0].subsections[0].items[0],  # First item
            structured_cv.sections[0].subsections[0].items[2],  # Third item
            structured_cv.sections[0].subsections[1].items[0],  # First item in second role
            structured_cv.sections[0].subsections[1].items[1],  # Second item in second role
        ]
        original_other_contents = [item.content for item in other_items]
        original_other_raw_outputs = [item.raw_llm_output for item in other_items]
        
        # Create agent state
        agent_state = AgentState(
            session_id=session_id,
            structured_cv=structured_cv,
            job_description_data=sample_job_description_data,
            current_item_id=str(target_item.id),
            current_section_key="Professional Experience",
            user_feedback="Please make this more specific with metrics",
            research_findings="Company values quantifiable achievements",
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock state manager
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager
            
            # Configure state manager mocks
            mock_state_manager.get_structured_cv.return_value = structured_cv
            mock_state_manager.update_item_status.return_value = True
            mock_state_manager.save_structured_cv.return_value = True
            
            # Execute single item processing
            result_state = await orchestrator.process_single_item(agent_state)
            
            # Validate processing success
            assert result_state is not None
            assert len(result_state.error_messages) == 0, f"Processing failed with errors: {result_state.error_messages}"
            
            # Validate target item was regenerated
            updated_cv = result_state.structured_cv
            updated_target_item = updated_cv.sections[0].subsections[0].items[1]
            
            assert updated_target_item.content != original_content, "Target item content should be regenerated"
            assert "REGENERATED:" in updated_target_item.content, "Target item should contain regenerated content"
            assert updated_target_item.raw_llm_output != original_raw_output, "Target item raw output should be updated"
            assert updated_target_item.status == ItemStatus.GENERATED, "Target item status should be GENERATED"
            assert updated_target_item.confidence_score == 0.92, "Target item should have updated confidence score"
            
            # Validate metadata was updated
            assert "regeneration_count" in updated_target_item.metadata
            assert updated_target_item.metadata["regeneration_count"] == 1
            
            # Validate other items were NOT modified (REQ-FUNC-GEN-3)
            updated_other_items = [
                updated_cv.sections[0].subsections[0].items[0],  # First item
                updated_cv.sections[0].subsections[0].items[2],  # Third item
                updated_cv.sections[0].subsections[1].items[0],  # First item in second role
                updated_cv.sections[0].subsections[1].items[1],  # Second item in second role
            ]
            
            for i, (original_content, original_raw, updated_item) in enumerate(zip(
                original_other_contents, original_other_raw_outputs, updated_other_items
            )):
                assert updated_item.content == original_content, f"Other item {i} content should be preserved"
                assert updated_item.raw_llm_output == original_raw, f"Other item {i} raw output should be preserved"
                assert updated_item.status != ItemStatus.TO_REGENERATE, f"Other item {i} should not be marked for regeneration"
            
            # Validate state manager interactions
            mock_state_manager.update_item_status.assert_called()
            mock_state_manager.save_structured_cv.assert_called()

    async def test_multiple_item_regeneration_workflow(self, orchestrator_with_content_writer_mock, sample_structured_cv_with_items, sample_job_description_data):
        """Test workflow with multiple individual item regenerations.
        
        Validates:
        - Sequential item processing
        - State consistency across multiple operations
        - User feedback integration
        """
        orchestrator = orchestrator_with_content_writer_mock
        session_id = str(uuid4())
        
        structured_cv = sample_structured_cv_with_items
        
        # Select multiple items to regenerate sequentially
        target_items = [
            structured_cv.sections[0].subsections[0].items[0],  # First item in first role
            structured_cv.sections[0].subsections[1].items[1],  # Second item in second role
            structured_cv.sections[1].items[0]  # First key qualification
        ]
        
        original_contents = [item.content for item in target_items]
        
        # Process each item individually
        current_cv = structured_cv
        for i, target_item in enumerate(target_items):
            # Create agent state for this item
            agent_state = AgentState(
                session_id=session_id,
                structured_cv=current_cv,
                job_description_data=sample_job_description_data,
                current_item_id=str(target_item.id),
                current_section_key="Professional Experience" if i < 2 else "Key Qualifications",
                user_feedback=f"Regeneration request {i+1}: Make this more specific",
                research_findings="Company values detailed achievements",
                error_messages=[],
                final_cv_output_path=None
            )
            
            # Mock state manager for this iteration
            with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
                mock_state_manager = MagicMock()
                mock_state_manager_class.return_value = mock_state_manager
                
                mock_state_manager.get_structured_cv.return_value = current_cv
                mock_state_manager.update_item_status.return_value = True
                mock_state_manager.save_structured_cv.return_value = True
                
                # Execute processing
                result_state = await orchestrator.process_single_item(agent_state)
                
                # Validate processing success
                assert result_state is not None
                assert len(result_state.error_messages) == 0
                
                # Update current CV for next iteration
                current_cv = result_state.structured_cv
        
        # Validate all target items were regenerated
        final_cv = current_cv
        
        # Check first role, first item
        updated_item_1 = final_cv.sections[0].subsections[0].items[0]
        assert updated_item_1.content != original_contents[0]
        assert "REGENERATED:" in updated_item_1.content
        
        # Check second role, second item
        updated_item_2 = final_cv.sections[0].subsections[1].items[1]
        assert updated_item_2.content != original_contents[1]
        assert "REGENERATED:" in updated_item_2.content
        
        # Check key qualification
        updated_item_3 = final_cv.sections[1].items[0]
        assert updated_item_3.content != original_contents[2]
        assert "REGENERATED:" in updated_item_3.content
        
        # Validate non-target items were preserved
        preserved_items = [
            final_cv.sections[0].subsections[0].items[1],  # Second item in first role
            final_cv.sections[0].subsections[0].items[2],  # Third item in first role
            final_cv.sections[0].subsections[1].items[0],  # First item in second role
            final_cv.sections[1].items[1]  # Second key qualification
        ]
        
        for item in preserved_items:
            assert "REGENERATED:" not in item.content, "Non-target items should not be regenerated"
            assert item.status == ItemStatus.GENERATED, "Non-target items should maintain original status"

    async def test_item_regeneration_with_user_feedback(self, orchestrator_with_content_writer_mock, sample_structured_cv_with_items, sample_job_description_data):
        """Test item regeneration incorporates user feedback.
        
        Validates:
        - REQ-FUNC-UI-3: User feedback integration
        - REQ-FUNC-UI-4: Feedback-driven content improvement
        """
        orchestrator = orchestrator_with_content_writer_mock
        session_id = str(uuid4())
        
        structured_cv = sample_structured_cv_with_items
        target_item = structured_cv.sections[0].subsections[0].items[0]
        
        # Test with specific user feedback
        user_feedback = "Add specific metrics and quantify the impact of the work"
        
        agent_state = AgentState(
            session_id=session_id,
            structured_cv=structured_cv,
            job_description_data=sample_job_description_data,
            current_item_id=str(target_item.id),
            current_section_key="Professional Experience",
            user_feedback=user_feedback,
            research_findings="Company emphasizes data-driven results",
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock state manager
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager
            
            mock_state_manager.get_structured_cv.return_value = structured_cv
            mock_state_manager.update_item_status.return_value = True
            mock_state_manager.save_structured_cv.return_value = True
            
            # Execute processing
            result_state = await orchestrator.process_single_item(agent_state)
            
            # Validate processing success
            assert result_state is not None
            assert len(result_state.error_messages) == 0
            
            # Validate content writer was called with correct parameters
            orchestrator.content_writer_agent.process_single_item.assert_called_once()
            call_args = orchestrator.content_writer_agent.process_single_item.call_args
            
            # Verify the call included the item ID and job data
            assert call_args[0][0] == target_item.id  # item_id
            assert call_args[0][1] == sample_job_description_data  # job_data
            assert call_args[0][2] == structured_cv  # cv_data
            
            # Validate user feedback was preserved in state
            assert result_state.user_feedback == user_feedback
            
            # Validate regenerated item contains enhanced content
            updated_item = result_state.structured_cv.sections[0].subsections[0].items[0]
            assert "REGENERATED:" in updated_item.content
            assert updated_item.metadata["regeneration_reason"] == "user_requested"

    async def test_item_regeneration_error_handling(self, sample_structured_cv_with_items, sample_job_description_data):
        """Test error handling during item regeneration.
        
        Validates:
        - REQ-NONFUNC-RELIABILITY-1: Graceful error handling
        - Error recovery mechanisms
        - State consistency during failures
        """
        session_id = str(uuid4())
        
        # Create orchestrator with failing content writer
        orchestrator = EnhancedOrchestrator()
        mock_content_writer = AsyncMock(spec=EnhancedContentWriterAgent)
        
        # Configure mock to simulate LLM failure
        async def mock_failing_process_single_item(item_id, job_data, cv_data, section_key=None):
            return {
                "success": False,
                "item": Item(
                    id=item_id,
                    content="⚠️ The LLM did not respond or the content was not correctly generated. Please wait 10 seconds and try to regenerate!",
                    status=ItemStatus.GENERATION_FAILED,
                    item_type=ItemType.BULLET_POINT,
                    confidence_score=0.0,
                    raw_llm_output="Error: LLM service timeout",
                    metadata={"error": "LLM service unavailable", "fallback_used": True}
                ),
                "metadata": {
                    "processing_time": 0.1,
                    "error_type": "llm_timeout",
                    "fallback_applied": True
                }
            }
        
        mock_content_writer.process_single_item = mock_failing_process_single_item
        orchestrator.content_writer_agent = mock_content_writer
        
        structured_cv = sample_structured_cv_with_items
        target_item = structured_cv.sections[0].subsections[0].items[0]
        original_content = target_item.content
        
        agent_state = AgentState(
            session_id=session_id,
            structured_cv=structured_cv,
            job_description_data=sample_job_description_data,
            current_item_id=str(target_item.id),
            current_section_key="Professional Experience",
            user_feedback="Regenerate this item",
            research_findings="Research data",
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock state manager
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager
            
            mock_state_manager.get_structured_cv.return_value = structured_cv
            mock_state_manager.update_item_status.return_value = True
            mock_state_manager.save_structured_cv.return_value = True
            
            # Execute processing (should handle error gracefully)
            result_state = await orchestrator.process_single_item(agent_state)
            
            # Validate error was handled gracefully
            assert result_state is not None
            # Note: Error messages might be present, but workflow should continue
            
            # Validate item status reflects the failure
            updated_item = result_state.structured_cv.sections[0].subsections[0].items[0]
            assert updated_item.status == ItemStatus.GENERATION_FAILED
            assert "⚠️" in updated_item.content  # Fallback content
            assert "error" in updated_item.metadata
            assert updated_item.metadata["fallback_used"] is True
            
            # Validate other items were not affected
            other_items = [
                result_state.structured_cv.sections[0].subsections[0].items[1],
                result_state.structured_cv.sections[0].subsections[0].items[2]
            ]
            
            for item in other_items:
                assert item.status != ItemStatus.GENERATION_FAILED
                assert "⚠️" not in item.content

    async def test_item_status_transitions(self, orchestrator_with_content_writer_mock, sample_structured_cv_with_items, sample_job_description_data):
        """Test proper item status transitions during processing.
        
        Validates:
        - Correct status flow: GENERATED -> TO_REGENERATE -> GENERATED
        - Status persistence
        - State consistency
        """
        orchestrator = orchestrator_with_content_writer_mock
        session_id = str(uuid4())
        
        structured_cv = sample_structured_cv_with_items
        target_item = structured_cv.sections[0].subsections[0].items[0]
        
        # Verify initial status
        assert target_item.status == ItemStatus.GENERATED
        
        # Mark item for regeneration (simulating UI action)
        target_item.status = ItemStatus.TO_REGENERATE
        
        agent_state = AgentState(
            session_id=session_id,
            structured_cv=structured_cv,
            job_description_data=sample_job_description_data,
            current_item_id=str(target_item.id),
            current_section_key="Professional Experience",
            user_feedback="Regenerate this item",
            research_findings="Research data",
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock state manager
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager
            
            mock_state_manager.get_structured_cv.return_value = structured_cv
            mock_state_manager.update_item_status.return_value = True
            mock_state_manager.save_structured_cv.return_value = True
            
            # Execute processing
            result_state = await orchestrator.process_single_item(agent_state)
            
            # Validate status transition
            updated_item = result_state.structured_cv.sections[0].subsections[0].items[0]
            assert updated_item.status == ItemStatus.GENERATED, "Item should return to GENERATED status after successful regeneration"
            
            # Validate state manager was called to update status
            mock_state_manager.update_item_status.assert_called()
            mock_state_manager.save_structured_cv.assert_called()