"""End-to-End test for complete CV generation workflow.

Tests the "happy path" of full CV generation from job description
to final tailored CV output, validating all functional requirements.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from uuid import uuid4

# Import application components
from src.orchestration.enhanced_orchestrator import EnhancedOrchestrator
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, Section, Subsection, Item, ItemStatus, ItemType,
    JobDescriptionData
)
from src.services.state_manager import StateManager
from src.services.llm import LLMService
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent


@pytest.mark.e2e
@pytest.mark.asyncio
class TestCompleteCVGeneration:
    """E2E tests for complete CV generation workflow."""

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return """
        Senior Software Engineer - AI/ML Platform

        We are seeking a Senior Software Engineer to join our AI/ML Platform team.

        Requirements:
        - 5+ years of software engineering experience
        - Strong Python programming skills
        - Experience with machine learning frameworks (TensorFlow, PyTorch)
        - Cloud platform experience (AWS, GCP, Azure)
        - Experience with containerization (Docker, Kubernetes)
        - Strong problem-solving and communication skills

        Responsibilities:
        - Design and implement scalable ML infrastructure
        - Collaborate with data scientists and ML engineers
        - Optimize model training and inference pipelines
        - Ensure system reliability and performance
        """

    @pytest.fixture
    def sample_base_cv(self):
        """Sample base CV content for testing."""
        return """
        John Doe
        Senior Software Engineer

        EXPERIENCE:
        Software Engineer @ TechCorp (2020-2023)
        - Developed web applications using Python and Django
        - Implemented REST APIs for mobile applications
        - Collaborated with cross-functional teams

        Junior Developer @ StartupXYZ (2018-2020)
        - Built frontend components using React
        - Participated in agile development processes
        - Contributed to code reviews and testing

        EDUCATION:
        Bachelor of Science in Computer Science
        University of Technology (2014-2018)

        SKILLS:
        Python, JavaScript, React, Django, PostgreSQL, Git
        """

    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for different agents."""
        return {
            "parser": {
                "job_description_data": {
                    "required_skills": ["Python", "Machine Learning", "Cloud Platforms"],
                    "responsibilities": ["Design ML infrastructure", "Optimize pipelines"],
                    "industry_terms": ["AI/ML", "TensorFlow", "PyTorch"],
                    "company_context": "AI/ML Platform team"
                },
                "structured_cv": {
                    "sections": [
                        {
                            "name": "Professional Experience",
                            "subsections": [
                                {
                                    "name": "Software Engineer @ TechCorp",
                                    "items": [
                                        {"content": "Developed web applications using Python and Django"},
                                        {"content": "Implemented REST APIs for mobile applications"}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            },
            "research": {
                "research_findings": "Company focuses on AI/ML platform development with emphasis on scalability and performance."
            },
            "content_writer": {
                "key_qualifications": [
                    "5+ years of software engineering experience with focus on AI/ML platforms",
                    "Expert in Python programming with machine learning frameworks",
                    "Proven experience with cloud platforms and containerization"
                ],
                "experience_items": [
                    "Architected and implemented scalable ML infrastructure using Python and TensorFlow, supporting 100+ concurrent model training jobs",
                    "Optimized model inference pipelines reducing latency by 40% through efficient containerization with Docker and Kubernetes"
                ]
            },
            "qa": {
                "quality_score": 0.85,
                "issues_found": [],
                "recommendations": ["Consider adding specific metrics to quantify achievements"]
            }
        }

    @pytest.fixture
    async def orchestrator_with_mocks(self, mock_llm_responses):
        """Create orchestrator with mocked dependencies."""
        # Mock LLM service
        mock_llm = AsyncMock(spec=LLMService)

        # Mock agents
        mock_parser = AsyncMock(spec=ParserAgent)
        mock_research = AsyncMock(spec=ResearchAgent)
        mock_content_writer = AsyncMock(spec=EnhancedContentWriterAgent)
        mock_qa = AsyncMock(spec=QualityAssuranceAgent)

        # Configure mock responses
        mock_parser.run_as_node.return_value = {
            "job_description_data": JobDescriptionData(**mock_llm_responses["parser"]["job_description_data"]),
            "structured_cv": StructuredCV(
                sections=[
                    Section(
                        name="Professional Experience",
                        subsections=[
                            Subsection(
                                name="Software Engineer @ TechCorp",
                                items=[
                                    Item(content="Developed web applications using Python and Django", status=ItemStatus.INITIAL),
                                    Item(content="Implemented REST APIs for mobile applications", status=ItemStatus.INITIAL)
                                ]
                            )
                        ]
                    )
                ]
            )
        }

        mock_research.run_as_node.return_value = {
            "research_findings": mock_llm_responses["research"]["research_findings"]
        }

        mock_content_writer.run_as_node.return_value = {
            "structured_cv": StructuredCV(
                sections=[
                    Section(
                        name="Professional Experience",
                        subsections=[
                            Subsection(
                                name="Software Engineer @ TechCorp",
                                items=[
                                    Item(
                                        content=mock_llm_responses["content_writer"]["experience_items"][0],
                                        status=ItemStatus.GENERATED,
                                        confidence_score=0.9
                                    ),
                                    Item(
                                        content=mock_llm_responses["content_writer"]["experience_items"][1],
                                        status=ItemStatus.GENERATED,
                                        confidence_score=0.85
                                    )
                                ]
                            )
                        ]
                    )
                ],
                big_10_skills=mock_llm_responses["content_writer"]["key_qualifications"]
            )
        }

        mock_qa.run_as_node.return_value = {
            "structured_cv": StructuredCV(
                sections=[
                    Section(
                        name="Professional Experience",
                        subsections=[
                            Subsection(
                                name="Software Engineer @ TechCorp",
                                items=[
                                    Item(
                                        content=mock_llm_responses["content_writer"]["experience_items"][0],
                                        status=ItemStatus.USER_ACCEPTED,
                                        confidence_score=0.9,
                                        metadata={"qa_score": 0.85, "qa_issues": []}
                                    ),
                                    Item(
                                        content=mock_llm_responses["content_writer"]["experience_items"][1],
                                        status=ItemStatus.USER_ACCEPTED,
                                        confidence_score=0.85,
                                        metadata={"qa_score": 0.85, "qa_issues": []}
                                    )
                                ]
                            )
                        ]
                    )
                ],
                big_10_skills=mock_llm_responses["content_writer"]["key_qualifications"]
            )
        }

        # Create orchestrator with mocked dependencies
        orchestrator = EnhancedOrchestrator()
        orchestrator.llm = mock_llm
        orchestrator.parser_agent = mock_parser
        orchestrator.research_agent = mock_research
        orchestrator.content_writer_agent = mock_content_writer
        orchestrator.qa_agent = mock_qa

        return orchestrator

    async def test_complete_cv_generation_happy_path(self, orchestrator_with_mocks, sample_job_description, sample_base_cv):
        """Test complete CV generation workflow - happy path.

        Validates:
        - REQ-FUNC-PARSE-1: Job description parsing
        - REQ-FUNC-GEN-1: Content generation
        - REQ-FUNC-GEN-2: Big 10 skills generation
        - REQ-FUNC-UI-6: Raw LLM output storage
        - REQ-NONFUNC-RELIABILITY-1: Error handling
        """
        orchestrator = orchestrator_with_mocks
        session_id = str(uuid4())

        # Create initial state
        initial_state = AgentState(
            session_id=session_id,
            job_description_raw=sample_job_description,
            base_cv_raw=sample_base_cv,
            structured_cv=None,
            job_description_data=None,
            current_item_id=None,
            current_section_key=None,
            user_feedback=None,
            research_findings=None,
            error_messages=[],
            final_cv_output_path=None
        )

        # Execute full workflow
        result_state = await orchestrator.execute_full_workflow(initial_state)

        # Validate workflow completion
        assert result_state is not None
        assert len(result_state.error_messages) == 0, f"Workflow failed with errors: {result_state.error_messages}"

        # Validate job description parsing (REQ-FUNC-PARSE-1)
        assert result_state.job_description_data is not None
        assert len(result_state.job_description_data.required_skills) > 0
        assert len(result_state.job_description_data.responsibilities) > 0

        # Validate structured CV generation (REQ-FUNC-GEN-1)
        assert result_state.structured_cv is not None
        assert len(result_state.structured_cv.sections) > 0

        # Find experience section
        experience_section = None
        for section in result_state.structured_cv.sections:
            if "experience" in section.name.lower():
                experience_section = section
                break

        assert experience_section is not None
        assert len(experience_section.subsections) > 0

        # Validate content generation and status updates
        for subsection in experience_section.subsections:
            assert len(subsection.items) > 0
            for item in subsection.items:
                # Items should be processed and accepted
                assert item.status == ItemStatus.USER_ACCEPTED
                assert item.confidence_score is not None
                assert item.confidence_score > 0.0

                # Validate QA metadata (REQ-NONFUNC-RELIABILITY-1)
                assert "qa_score" in item.metadata
                assert "qa_issues" in item.metadata

        # Validate Big 10 skills generation (REQ-FUNC-GEN-2)
        assert result_state.structured_cv.big_10_skills is not None
        assert len(result_state.structured_cv.big_10_skills) > 0

        # Validate research findings
        assert result_state.research_findings is not None
        assert len(result_state.research_findings) > 0

        # Validate agent execution order
        orchestrator.parser_agent.run_as_node.assert_called_once()
        orchestrator.research_agent.run_as_node.assert_called_once()
        orchestrator.content_writer_agent.run_as_node.assert_called_once()
        orchestrator.qa_agent.run_as_node.assert_called_once()

    async def test_complete_cv_generation_with_state_persistence(self, orchestrator_with_mocks, sample_job_description, sample_base_cv):
        """Test CV generation with state persistence.

        Validates:
        - State manager integration
        - Session persistence
        - Data consistency across workflow steps
        """
        orchestrator = orchestrator_with_mocks
        session_id = str(uuid4())

        # Mock state manager
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager

            # Configure state manager mocks
            mock_state_manager.get_structured_cv.return_value = None
            mock_state_manager.save_structured_cv.return_value = True
            mock_state_manager.update_item_status.return_value = True

            # Create initial state
            initial_state = AgentState(
                session_id=session_id,
                job_description_raw=sample_job_description,
                base_cv_raw=sample_base_cv,
                structured_cv=None,
                job_description_data=None,
                current_item_id=None,
                current_section_key=None,
                user_feedback=None,
                research_findings=None,
                error_messages=[],
                final_cv_output_path=None
            )

            # Execute workflow
            result_state = await orchestrator.execute_full_workflow(initial_state)

            # Validate state persistence calls
            assert result_state is not None
            assert len(result_state.error_messages) == 0

            # Verify state manager was used for persistence
            mock_state_manager_class.assert_called_with(session_id)

    async def test_workflow_performance_requirements(self, orchestrator_with_mocks, sample_job_description, sample_base_cv):
        """Test workflow meets performance requirements.

        Validates:
        - REQ-NONFUNC-PERF-1: Response time under 30 seconds
        - REQ-NONFUNC-PERF-2: Concurrent user support
        """
        import time

        orchestrator = orchestrator_with_mocks
        session_id = str(uuid4())

        # Create initial state
        initial_state = AgentState(
            session_id=session_id,
            job_description_raw=sample_job_description,
            base_cv_raw=sample_base_cv,
            structured_cv=None,
            job_description_data=None,
            current_item_id=None,
            current_section_key=None,
            user_feedback=None,
            research_findings=None,
            error_messages=[],
            final_cv_output_path=None
        )

        # Measure execution time
        start_time = time.time()
        result_state = await orchestrator.execute_full_workflow(initial_state)
        end_time = time.time()

        execution_time = end_time - start_time

        # Validate performance requirement (REQ-NONFUNC-PERF-1)
        assert execution_time < 30.0, f"Workflow took {execution_time:.2f} seconds, exceeding 30-second limit"

        # Validate successful completion
        assert result_state is not None
        assert len(result_state.error_messages) == 0

        print(f"Workflow completed in {execution_time:.2f} seconds")

    async def test_concurrent_workflow_execution(self, orchestrator_with_mocks, sample_job_description, sample_base_cv):
        """Test concurrent workflow execution.

        Validates:
        - REQ-NONFUNC-PERF-2: Support for multiple concurrent users
        - Session isolation
        - Resource management
        """
        orchestrator = orchestrator_with_mocks

        # Create multiple concurrent workflows
        async def run_workflow(session_suffix: str):
            session_id = f"concurrent_test_{session_suffix}"
            initial_state = AgentState(
                session_id=session_id,
                job_description_raw=sample_job_description,
                base_cv_raw=sample_base_cv,
                structured_cv=None,
                job_description_data=None,
                current_item_id=None,
                current_section_key=None,
                user_feedback=None,
                research_findings=None,
                error_messages=[],
                final_cv_output_path=None
            )

            result_state = await orchestrator.execute_full_workflow(initial_state)
            return result_state

        # Run 3 concurrent workflows
        tasks = [
            run_workflow("1"),
            run_workflow("2"),
            run_workflow("3")
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate all workflows completed successfully
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Workflow {i+1} failed with exception: {result}"
            assert result is not None, f"Workflow {i+1} returned None"
            assert len(result.error_messages) == 0, f"Workflow {i+1} had errors: {result.error_messages}"

            # Validate session isolation
            expected_session_id = f"concurrent_test_{i+1}"
            assert result.session_id == expected_session_id, f"Session ID mismatch for workflow {i+1}"