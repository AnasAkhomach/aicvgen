"""Unit tests for PDF generation functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.agents.formatter_agent import FormatterAgent
from src.orchestration.state import AgentState
from src.models.data_models import StructuredCV, Section, Item


class TestFormatterAgent:
    """Test cases for FormatterAgent PDF generation."""

    @pytest.fixture
    def formatter_agent(self):
        """Create a FormatterAgent instance for testing."""
        return FormatterAgent(
            name="TestFormatterAgent",
            description="Test formatter agent",
            llm_service=Mock(),
            error_recovery_service=Mock(),
            progress_tracker=Mock(),
        )

    @pytest.fixture
    def sample_cv_data(self):
        """Create sample CV data for testing."""
        return StructuredCV(
            metadata={
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123",
                "linkedin": "https://linkedin.com/in/johndoe",
            },
            sections=[
                Section(
                    name="Key Qualifications",
                    items=[
                        Item(
                            content="Python Programming", item_type="key_qualification"
                        ),
                        Item(content="Machine Learning", item_type="key_qualification"),
                        Item(content="Data Analysis", item_type="key_qualification"),
                    ],
                ),
                Section(
                    name="Professional Experience",
                    items=[
                        Item(
                            content="Led development of AI-powered applications",
                            item_type="bullet_point",
                        ),
                        Item(
                            content="Managed team of 5 engineers",
                            item_type="bullet_point",
                        ),
                    ],
                ),
            ],
        )

    @pytest.fixture
    def sample_state(self, sample_cv_data):
        """Create sample AgentState for testing."""
        from src.models.data_models import JobDescriptionData

        job_data = JobDescriptionData(
            raw_text="Sample job description for testing",
            skills=["Python", "Machine Learning"],
            experience_level="Senior",
            responsibilities=["Develop software", "Lead team"],
            industry_terms=["AI", "Software Development"],
            company_values=["Innovation", "Collaboration"],
        )

        return AgentState(
            structured_cv=sample_cv_data,
            job_description_data=job_data,
            error_messages=[],
        )

    @pytest.mark.asyncio
    @patch("src.agents.formatter_agent.Environment")
    @patch("src.agents.formatter_agent.HTML")
    async def test_run_as_node_success(
        self,
        mock_html,
        mock_env,
        formatter_agent,
        sample_state,
    ):
        """Test successful PDF generation."""
        # Setup mocks
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test CV</html>"
        mock_jinja_env = Mock()
        mock_jinja_env.get_template.return_value = mock_template
        mock_env.return_value = mock_jinja_env

        mock_html_instance = Mock()
        mock_html_instance.write_pdf.return_value = b"fake pdf content"
        mock_html.return_value = mock_html_instance

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "data" / "output"
            # Patch project_root if needed in your implementation
            # mock_config.project_root = Path(temp_dir)

            # Execute
            result = await formatter_agent.run_as_node(sample_state)

            # Verify
            assert "final_output_path" in result
            assert "CV_" in result["final_output_path"]
            assert ".pdf" in result["final_output_path"]
            assert "error_messages" not in result

            # Verify template rendering was called with correct data
            mock_template.render.assert_called_once_with(cv=sample_state.structured_cv)

            # Verify PDF generation was called
            mock_html_instance.write_pdf.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.agents.formatter_agent.logger")
    async def test_run_as_node_no_cv_data(self, mock_logger, formatter_agent):
        """Test handling when no CV data is provided."""
        from src.models.data_models import JobDescriptionData, StructuredCV

        job_data = JobDescriptionData(raw_text="Test job description")
        # Create an empty StructuredCV to satisfy validation but simulate no data
        empty_cv = StructuredCV(
            cv_id="empty-cv",
            personal_info={},
            professional_experience=[],
            education=[],
            skills=[],
            side_projects=[],
        )
        state = AgentState(
            structured_cv=empty_cv, job_description_data=job_data, error_messages=[]
        )

        result = await formatter_agent.run_as_node(state)

        assert "error_messages" in result
        assert "No CV data found in state" in result["error_messages"][0]
        assert "final_output_path" not in result

    @pytest.mark.asyncio
    @patch("src.agents.formatter_agent.logger")
    @patch("src.agents.formatter_agent.Environment")
    async def test_run_as_node_template_error(
        self, mock_env, mock_logger, formatter_agent, sample_state
    ):
        """Test handling of template rendering errors."""
        # Setup mocks
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test CV</html>"
        mock_jinja_env = Mock()
        mock_jinja_env.get_template.side_effect = Exception("Template not found")
        mock_env.return_value = mock_jinja_env

        # Execute
        result = await formatter_agent.run_as_node(sample_state)

        # Verify error handling
        assert "error_messages" in result
        assert "PDF generation failed" in result["error_messages"][0]
        assert "final_output_path" not in result

    @pytest.mark.asyncio
    @patch("src.agents.formatter_agent.logger")
    @patch("src.agents.formatter_agent.Environment")
    async def test_run_as_node_css_file_missing(
        self, mock_env, mock_logger, formatter_agent, sample_state
    ):
        """Test handling when CSS file is missing."""
        # Setup mocks
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test CV</html>"
        mock_jinja_env = Mock()
        mock_jinja_env.get_template.return_value = mock_template
        mock_env.return_value = mock_jinja_env

        # Create temporary directory without CSS file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch project_root if needed in your implementation
            # mock_config.project_root = Path(temp_dir)

            # Execute
            result = await formatter_agent.run_as_node(sample_state)

            # Verify it still works without CSS
            assert "final_output_path" in result
            # Verify PDF generation was called with no stylesheets
            mock_html_instance.write_pdf.assert_called_once_with(stylesheets=None)
