"""E2E integration test for full user session workflow.

Tests the complete user-in-the-loop workflow including:
- Initial CV generation
- User feedback (accept/regenerate)
- Iterative refinement
- Final output generation
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
from uuid import uuid4

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.core.state_manager import StateManager
from src.models.workflow_models import AgentState, UserFeedback, UserAction
from src.models.data_models import (
    StructuredCV, Section, Item, ItemStatus, ItemType,
    JobDescriptionData
)
from src.orchestration.cv_workflow_graph import get_cv_workflow_graph
from tests.e2e.mock_llm_service import MockLLMService


@pytest.mark.e2e
@pytest.mark.asyncio
class TestFullUserSession:
    """E2E test for complete user session workflow."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service with realistic responses."""
        service = MockLLMService()
        
        # Configure realistic responses for different stages
        service.add_response_pattern(
            "experience",
            "Senior Software Engineer at TechCorp (2020-2023)\n• Led development of microservices architecture\n• Managed team of 5 developers\n• Implemented CI/CD pipelines"
        )
        service.add_response_pattern(
            "education", 
            "Master of Science in Computer Science\nStanford University (2018-2020)\n• GPA: 3.8/4.0\n• Thesis: Machine Learning for Distributed Systems"
        )
        service.add_response_pattern(
            "skills",
            "Python, Java, AWS, Docker, Kubernetes, React, Node.js, PostgreSQL, Redis, Git"
        )
        
        return service

    @pytest.fixture
    def state_manager(self, tmp_path):
        """Create a state manager for testing."""
        return StateManager(session_id="test_session_123")

    @pytest.fixture
    def orchestrator(self, state_manager, mock_llm_service):
        """Create an orchestrator with mocked dependencies."""
        with patch('src.core.enhanced_orchestrator.get_cv_workflow_graph') as mock_graph:
            # Mock the workflow graph
            mock_app = AsyncMock()
            mock_graph.return_value = mock_app
            
            orchestrator = EnhancedOrchestrator(
                state_manager=state_manager,
                llm_service=mock_llm_service
            )
            orchestrator.workflow_app = mock_app
            return orchestrator

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return """
        Senior Software Engineer - Full Stack
        
        We are looking for a Senior Software Engineer with:
        - 5+ years of experience in full-stack development
        - Proficiency in Python, JavaScript, and cloud technologies
        - Experience with microservices and containerization
        - Strong leadership and mentoring skills
        
        Responsibilities:
        - Lead development of scalable web applications
        - Mentor junior developers
        - Design and implement microservices architecture
        - Collaborate with product and design teams
        """

    @pytest.fixture
    def sample_base_cv(self):
        """Sample base CV structure for testing."""
        return StructuredCV(
            sections={
                "experience": Section(
                    title="Professional Experience",
                    items={
                        "exp_1": Item(
                            id="exp_1",
                            content="Software Engineer at TechCorp",
                            status=ItemStatus.PENDING
                        ),
                        "exp_2": Item(
                            id="exp_2",
                            content="Junior Developer at StartupXYZ",
                            status=ItemStatus.PENDING
                        )
                    }
                ),
                "education": Section(
                    title="Education",
                    items={
                        "edu_1": Item(
                            id="edu_1",
                            content="BS Computer Science",
                            status=ItemStatus.PENDING
                        )
                    }
                ),
                "skills": Section(
                    title="Technical Skills",
                    items={
                        "skill_1": Item(
                            id="skill_1",
                            content="Python, JavaScript",
                            status=ItemStatus.PENDING
                        )
                    }
                )
            }
        )

    async def test_complete_user_session_workflow(self, orchestrator, state_manager, 
                                                 sample_job_description, sample_base_cv):
        """Test the complete user session workflow from start to finish."""
        
        # Phase 1: Initial Setup
        job_data = JobDescriptionData(
            raw_text=sample_job_description,
            requirements=["Python", "JavaScript", "Microservices"],
            responsibilities=["Lead development", "Mentor developers"]
        )
        
        state_manager.set_job_description_data(job_data)
        state_manager.set_structured_cv(sample_base_cv)
        
        # Mock the workflow execution to return processed state
        def mock_workflow_execution(state_dict):
            state = AgentState.model_validate(state_dict)
            # Simulate processing the first item
            if state.current_item_id == "exp_1":
                # Update the item as generated
                updated_cv = state.structured_cv.model_copy(deep=True)
                updated_cv.sections["experience"].items["exp_1"].status = ItemStatus.GENERATED
                updated_cv.sections["experience"].items["exp_1"].content = "Enhanced: Senior Software Engineer at TechCorp"
                
                return {
                    **state_dict,
                    "structured_cv": updated_cv.model_dump(),
                    "current_item_id": "exp_2",
                    "items_to_process_queue": ["exp_2"]
                }
            return state_dict
        
        orchestrator.workflow_app.ainvoke = AsyncMock(side_effect=mock_workflow_execution)
        
        # Phase 2: Process First Item
        result_state = await orchestrator.process_single_item("exp_1")
        
        # Verify first item was processed
        assert result_state.structured_cv is not None
        exp_item = result_state.structured_cv.sections["experience"].items["exp_1"]
        assert exp_item.status == ItemStatus.GENERATED
        assert "Enhanced:" in exp_item.content
        
        # Phase 3: User Provides Regenerate Feedback
        user_feedback = UserFeedback(
            item_id="exp_1",
            action=UserAction.REGENERATE,
            feedback_text="Please make this more detailed and include specific technologies"
        )
        
        # Store feedback in state manager
        state_manager.add_user_feedback(user_feedback)
        
        # Mock regeneration workflow
        def mock_regeneration_execution(state_dict):
            state = AgentState.model_validate(state_dict)
            if state.current_item_id == "exp_1" and state.user_feedback:
                # Simulate regeneration with user feedback
                updated_cv = state.structured_cv.model_copy(deep=True)
                updated_cv.sections["experience"].items["exp_1"].content = (
                    "Enhanced with feedback: Senior Software Engineer at TechCorp (2020-2023) - "
                    "Led development of microservices using Python and Docker, managed CI/CD pipelines"
                )
                
                return {
                    **state_dict,
                    "structured_cv": updated_cv.model_dump(),
                    "user_feedback": []  # Clear feedback after processing
                }
            return state_dict
        
        orchestrator.workflow_app.ainvoke = AsyncMock(side_effect=mock_regeneration_execution)
        
        # Phase 4: Process Regeneration
        regenerated_state = await orchestrator.process_single_item("exp_1")
        
        # Verify regeneration incorporated feedback
        regenerated_item = regenerated_state.structured_cv.sections["experience"].items["exp_1"]
        assert "Enhanced with feedback:" in regenerated_item.content
        assert "microservices" in regenerated_item.content.lower()
        assert "docker" in regenerated_item.content.lower()
        
        # Phase 5: User Accepts Item
        accept_feedback = UserFeedback(
            item_id="exp_1",
            action=UserAction.ACCEPT,
            feedback_text="This looks great now!"
        )
        
        state_manager.add_user_feedback(accept_feedback)
        
        # Phase 6: Continue with Next Item
        def mock_next_item_execution(state_dict):
            state = AgentState.model_validate(state_dict)
            if state.current_item_id == "exp_2":
                updated_cv = state.structured_cv.model_copy(deep=True)
                updated_cv.sections["experience"].items["exp_2"].status = ItemStatus.GENERATED
                updated_cv.sections["experience"].items["exp_2"].content = "Enhanced: Junior Developer at StartupXYZ"
                
                return {
                    **state_dict,
                    "structured_cv": updated_cv.model_dump(),
                    "items_to_process_queue": [],
                    "current_section_key": "education"
                }
            return state_dict
        
        orchestrator.workflow_app.ainvoke = AsyncMock(side_effect=mock_next_item_execution)
        
        next_item_state = await orchestrator.process_single_item("exp_2")
        
        # Verify second item was processed
        exp_2_item = next_item_state.structured_cv.sections["experience"].items["exp_2"]
        assert exp_2_item.status == ItemStatus.GENERATED
        assert "Enhanced:" in exp_2_item.content
        
        # Phase 7: Verify Session State Consistency
        final_cv = state_manager.get_structured_cv()
        assert final_cv is not None
        
        # Check that all processed items are properly updated
        assert final_cv.sections["experience"].items["exp_1"].status == ItemStatus.GENERATED
        assert final_cv.sections["experience"].items["exp_2"].status == ItemStatus.GENERATED
        
        # Verify feedback was processed and cleared
        assert len(state_manager.get_user_feedback()) == 0  # Should be cleared after processing

    async def test_user_session_with_multiple_regenerations(self, orchestrator, state_manager,
                                                           sample_job_description, sample_base_cv):
        """Test user session with multiple regeneration cycles."""
        
        # Setup initial state
        job_data = JobDescriptionData(
            raw_text=sample_job_description,
            requirements=["Python", "Leadership"],
            responsibilities=["Lead development"]
        )
        
        state_manager.set_job_description_data(job_data)
        state_manager.set_structured_cv(sample_base_cv)
        
        # Track regeneration attempts
        regeneration_count = 0
        
        def mock_multiple_regenerations(state_dict):
            nonlocal regeneration_count
            state = AgentState.model_validate(state_dict)
            
            if state.current_item_id == "exp_1":
                regeneration_count += 1
                updated_cv = state.structured_cv.model_copy(deep=True)
                
                # Different content based on regeneration attempt
                if regeneration_count == 1:
                    content = "First attempt: Software Engineer at TechCorp"
                elif regeneration_count == 2:
                    content = "Second attempt: Senior Software Engineer at TechCorp with leadership experience"
                else:
                    content = "Final attempt: Senior Software Engineer at TechCorp - Led team of 5 developers"
                
                updated_cv.sections["experience"].items["exp_1"].content = content
                updated_cv.sections["experience"].items["exp_1"].status = ItemStatus.GENERATED
                
                return {
                    **state_dict,
                    "structured_cv": updated_cv.model_dump(),
                    "user_feedback": []
                }
            return state_dict
        
        orchestrator.workflow_app.ainvoke = AsyncMock(side_effect=mock_multiple_regenerations)
        
        # First generation
        await orchestrator.process_single_item("exp_1")
        assert regeneration_count == 1
        
        # First regeneration
        feedback_1 = UserFeedback(
            item_id="exp_1",
            action=UserAction.REGENERATE,
            feedback_text="Add more leadership details"
        )
        state_manager.add_user_feedback(feedback_1)
        await orchestrator.process_single_item("exp_1")
        assert regeneration_count == 2
        
        # Second regeneration
        feedback_2 = UserFeedback(
            item_id="exp_1",
            action=UserAction.REGENERATE,
            feedback_text="Include team size"
        )
        state_manager.add_user_feedback(feedback_2)
        await orchestrator.process_single_item("exp_1")
        assert regeneration_count == 3
        
        # Verify final content includes all requested details
        final_cv = state_manager.get_structured_cv()
        final_content = final_cv.sections["experience"].items["exp_1"].content
        assert "Led team of 5 developers" in final_content

    async def test_user_session_error_recovery(self, orchestrator, state_manager,
                                              sample_job_description, sample_base_cv):
        """Test user session with error recovery."""
        
        # Setup initial state
        job_data = JobDescriptionData(
            raw_text=sample_job_description,
            requirements=["Python"],
            responsibilities=["Development"]
        )
        
        state_manager.set_job_description_data(job_data)
        state_manager.set_structured_cv(sample_base_cv)
        
        # Mock workflow that fails first, then succeeds
        attempt_count = 0
        
        def mock_error_recovery(state_dict):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count == 1:
                # First attempt fails
                raise Exception("Simulated LLM service error")
            else:
                # Second attempt succeeds
                state = AgentState.model_validate(state_dict)
                updated_cv = state.structured_cv.model_copy(deep=True)
                updated_cv.sections["experience"].items["exp_1"].content = "Recovered: Software Engineer at TechCorp"
                updated_cv.sections["experience"].items["exp_1"].status = ItemStatus.GENERATED
                
                return {
                    **state_dict,
                    "structured_cv": updated_cv.model_dump()
                }
        
        orchestrator.workflow_app.ainvoke = AsyncMock(side_effect=mock_error_recovery)
        
        # First attempt should handle error gracefully
        result_state = await orchestrator.process_single_item("exp_1")
        
        # Verify error was handled
        assert attempt_count == 1
        item_status = state_manager.get_item_status("exp_1")
        assert item_status == ItemStatus.GENERATION_FAILED
        
        # Second attempt should succeed
        result_state = await orchestrator.process_single_item("exp_1")
        
        # Verify recovery
        assert attempt_count == 2
        final_cv = state_manager.get_structured_cv()
        recovered_item = final_cv.sections["experience"].items["exp_1"]
        assert recovered_item.status == ItemStatus.GENERATED
        assert "Recovered:" in recovered_item.content