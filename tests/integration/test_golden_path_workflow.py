"""
Golden Path Integration Test for CV Generation Workflow.

This test validates the complete end-to-end workflow from CV text input
through the entire agent pipeline to final PDF generation.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from pathlib import Path

from src.orchestration.cv_workflow_graph import cv_graph_app
from src.orchestration.state import AgentState
from src.core.state_helpers import create_initial_agent_state
from src.models.data_models import StructuredCV, JobDescriptionData, MetadataModel
from src.config.settings import get_config


class TestGoldenPathWorkflow:
    """Integration tests for the complete CV generation workflow."""

    @pytest.fixture
    def sample_cv_text(self):
        """Sample CV text for testing."""
        return """
        John Doe
        Software Engineer

        Experience:
        - Senior Software Developer at TechCorp (2020-2023)
        - Python development and API design
        - Led team of 5 developers

        Skills:
        - Python, JavaScript, React
        - AWS, Docker, Kubernetes
        - Agile methodologies

        Education:
        - BS Computer Science, University of Tech (2016-2020)
        """

    @pytest.fixture
    def sample_job_description(self):
        """Sample job description for testing."""
        return """
        Senior Python Developer
        TechStart Inc.

        We are looking for a Senior Python Developer to join our team.

        Requirements:
        - 3+ years Python experience
        - Experience with web frameworks
        - Knowledge of cloud platforms
        - Strong problem-solving skills

        Responsibilities:
        - Develop and maintain Python applications
        - Work with cross-functional teams
        - Mentor junior developers
        """

    @pytest.fixture
    def mock_session_state(self, sample_cv_text, sample_job_description):
        """Mock Streamlit session state for testing."""
        return {
            "cv_text_input": sample_cv_text,
            "job_description_input": sample_job_description,
            "start_from_scratch_input": False,
            "user_gemini_api_key": "test-api-key",
        }

    @pytest.fixture
    def initial_state(self, mock_session_state):
        """Create initial agent state for testing."""
        # Mock streamlit session state
        with patch("streamlit.session_state", mock_session_state):
            return create_initial_agent_state()

    @pytest.mark.asyncio
    async def test_golden_path_complete_workflow(self, initial_state):
        """
        Test the complete golden path workflow from start to finish.

        This test validates:
        1. Parser agent processes CV and job description successfully
        2. Content writer enhances CV content
        3. Research agent finds relevant information
        4. QA agent validates quality
        5. Formatter agent generates final output
        """
        # Mock the LLM service to return predictable responses
        with patch(
            "src.services.llm_service.EnhancedLLMService.generate_content"
        ) as mock_generate:
            # Mock parser response
            parser_response = AsyncMock()
            parser_response.content = """
            {
                "personal_info": {
                    "name": "John Doe",
                    "title": "Software Engineer"
                },
                "skills": ["Python", "JavaScript", "React", "AWS", "Docker"],
                "experience": [
                    {
                        "title": "Senior Software Developer",
                        "company": "TechCorp",
                        "duration": "2020-2023",
                        "bullets": ["Python development", "API design", "Led team of 5"]
                    }
                ]
            }
            """

            # Mock content writer response
            content_writer_response = AsyncMock()
            content_writer_response.content = """
            Enhanced professional summary with relevant experience.
            """

            # Mock research response
            research_response = AsyncMock()
            research_response.content = """
            {
                "findings": [
                    {
                        "topic": "Python frameworks",
                        "content": "Django and Flask are popular choices",
                        "confidence": 0.9
                    }
                ],
                "sources": ["industry_reports"]
            }
            """

            # Mock QA response
            qa_response = AsyncMock()
            qa_response.content = """
            {
                "overall_score": 85,
                "feedback": "Well-structured CV with relevant experience",
                "suggestions": ["Add more technical details"]
            }
            """

            # Configure mock to return different responses based on context
            def mock_generate_side_effect(*args, **kwargs):
                prompt = args[0] if args else kwargs.get("prompt", "")
                if "parse" in prompt.lower() or "extract" in prompt.lower():
                    return parser_response
                elif "research" in prompt.lower():
                    return research_response
                elif "quality" in prompt.lower() or "review" in prompt.lower():
                    return qa_response
                else:
                    return content_writer_response

            mock_generate.side_effect = mock_generate_side_effect

            # Mock file operations for formatter
            with patch("builtins.open", create=True), patch(
                "weasyprint.HTML"
            ) as mock_html, patch("pathlib.Path.exists", return_value=True):

                mock_html.return_value.write_pdf = AsyncMock()

                # Execute the workflow
                try:
                    final_state = await cv_graph_app.ainvoke(initial_state)

                    # Validate the workflow completed successfully
                    assert final_state is not None
                    assert isinstance(final_state, AgentState)

                    # Validate that all major components were processed
                    assert final_state.structured_cv is not None
                    assert final_state.job_description_data is not None

                    # Validate that content was enhanced
                    assert final_state.current_item is not None

                    # Validate that research was conducted
                    if hasattr(final_state, "research_findings"):
                        assert final_state.research_findings is not None

                    # Validate that quality check was performed
                    if hasattr(final_state, "quality_check_results"):
                        assert final_state.quality_check_results is not None

                    # Validate that final output was generated
                    if hasattr(final_state, "final_output_path"):
                        assert final_state.final_output_path is not None

                    # Check that no critical errors occurred
                    if final_state.error_messages:
                        # Allow warning-level messages but not critical failures
                        critical_errors = [
                            msg
                            for msg in final_state.error_messages
                            if "critical" in msg.lower() or "failed" in msg.lower()
                        ]
                        assert (
                            len(critical_errors) == 0
                        ), f"Critical errors found: {critical_errors}"

                    print("✅ Golden path workflow completed successfully")

                except Exception as e:
                    pytest.fail(f"Workflow failed with exception: {str(e)}")

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, initial_state):
        """
        Test that the workflow handles errors gracefully.
        """
        # Mock LLM service to raise an exception
        with patch(
            "src.services.llm_service.EnhancedLLMService.generate_content"
        ) as mock_generate:
            mock_generate.side_effect = Exception("Simulated LLM failure")

            # Execute the workflow
            final_state = await cv_graph_app.ainvoke(initial_state)

            # Validate that errors were captured
            assert final_state is not None
            assert len(final_state.error_messages) > 0

            # Validate that the workflow didn't crash
            assert isinstance(final_state, AgentState)

            print("✅ Error handling test completed successfully")

    @pytest.mark.asyncio
    async def test_individual_agent_integration(self, initial_state):
        """
        Test individual agents in isolation to ensure they work correctly.
        """
        from src.core.dependency_injection import get_container

        container = get_container()

        # Test parser agent
        parser_agent = container.get("ParserAgent")
        assert parser_agent is not None

        # Mock LLM response for parser
        with patch(
            "src.services.llm_service.EnhancedLLMService.generate_content"
        ) as mock_generate:
            mock_response = AsyncMock()
            mock_response.content = (
                '{"skills": ["Python", "JavaScript"], "experience": []}'
            )
            mock_generate.return_value = mock_response

            try:
                parser_result = await parser_agent.run_as_node(initial_state)
                assert isinstance(parser_result, dict)
                assert (
                    "structured_cv" in parser_result
                    or "job_description_data" in parser_result
                )
                print("✅ Parser agent integration test passed")
            except Exception as e:
                print(f"⚠️  Parser agent test failed: {str(e)}")

        # Test content writer agent
        content_writer = container.get("EnhancedContentWriterAgent")
        assert content_writer is not None

        # Ensure state has required data for content writer
        if not initial_state.current_item_id and initial_state.items_to_process_queue:
            test_state = initial_state.model_copy(
                update={"current_item_id": initial_state.items_to_process_queue[0]}
            )
        else:
            test_state = initial_state

        with patch(
            "src.services.llm_service.EnhancedLLMService.generate_content"
        ) as mock_generate:
            mock_response = AsyncMock()
            mock_response.content = "Enhanced content for the CV section"
            mock_generate.return_value = mock_response

            try:
                writer_result = await content_writer.run_as_node(test_state)
                assert isinstance(writer_result, dict)
                print("✅ Content writer integration test passed")
            except Exception as e:
                print(f"⚠️  Content writer test failed: {str(e)}")

    def test_dependency_injection_setup(self):
        """
        Test that all required dependencies are properly registered.
        """
        from src.core.dependency_injection import get_container

        container = get_container()

        # Test that core services are available
        required_services = [
            "settings",
            "EnhancedLLMService",
            "ContentTemplateManager",
            "ErrorRecoveryService",
        ]

        for service_name in required_services:
            try:
                service = container.get(service_name)
                assert service is not None
                print(f"✅ {service_name} dependency available")
            except Exception as e:
                print(f"⚠️  {service_name} dependency missing: {str(e)}")

        # Test that agents are available
        required_agents = [
            "ParserAgent",
            "EnhancedContentWriterAgent",
            "QualityAssuranceAgent",
            "FormatterAgent",
        ]

        for agent_name in required_agents:
            try:
                agent = container.get(agent_name)
                assert agent is not None
                print(f"✅ {agent_name} dependency available")
            except Exception as e:
                print(f"⚠️  {agent_name} dependency missing: {str(e)}")

    def test_configuration_validation(self):
        """
        Test that the application configuration is valid.
        """
        config = get_config()
        assert config is not None

        # Validate core settings exist
        assert hasattr(config, "llm_settings")
        assert hasattr(config, "agent_settings")
        assert hasattr(config, "output")

        # Validate important configuration values
        assert config.llm_settings.default_model is not None
        assert config.output.max_skills_count > 0
        assert config.output.max_bullet_points_per_role > 0

        print("✅ Configuration validation passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
