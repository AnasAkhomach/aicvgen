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
            description="Test formatter agent"
        )

    @pytest.fixture
    def sample_cv_data(self):
        """Create sample CV data for testing."""
        return StructuredCV(
            metadata={
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123",
                "linkedin": "https://linkedin.com/in/johndoe"
            },
            sections=[
                Section(
                    name="Key Qualifications",
                    items=[
                        Item(content="Python Programming", item_type="key_qualification"),
                        Item(content="Machine Learning", item_type="key_qualification"),
                        Item(content="Data Analysis", item_type="key_qualification")
                    ]
                ),
                Section(
                    name="Professional Experience",
                    items=[
                        Item(content="Led development of AI-powered applications", item_type="bullet_point"),
                        Item(content="Managed team of 5 engineers", item_type="bullet_point")
                    ]
                )
            ]
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
            company_values=["Innovation", "Collaboration"]
        )
        
        return AgentState(
            structured_cv=sample_cv_data,
            job_description_data=job_data,
            error_messages=[]
        )

    @pytest.mark.asyncio
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.Environment')
    @patch('src.agents.formatter_agent.HTML')
    @patch('src.agents.formatter_agent.CSS')
    async def test_run_as_node_success(self, mock_css, mock_html, mock_env, mock_get_config, 
                                 formatter_agent, sample_state):
        """Test successful PDF generation."""
        # Setup mocks
        mock_config = Mock()
        mock_config.project_root = Path("/test/project")
        mock_get_config.return_value = mock_config
        
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test CV</html>"
        mock_jinja_env = Mock()
        mock_jinja_env.get_template.return_value = mock_template
        mock_env.return_value = mock_jinja_env
        
        mock_css_instance = Mock()
        mock_css.return_value = mock_css_instance
        
        mock_html_instance = Mock()
        mock_html_instance.write_pdf.return_value = b"fake pdf content"
        mock_html.return_value = mock_html_instance
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "data" / "output"
            mock_config.project_root = Path(temp_dir)
            
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
    @patch('src.agents.formatter_agent.logger')
    @patch('src.agents.formatter_agent.get_config')
    async def test_run_as_node_no_cv_data(self, mock_get_config, mock_logger, formatter_agent):
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
            side_projects=[]
        )
        state = AgentState(structured_cv=empty_cv, job_description_data=job_data, error_messages=[])
        
        result = await formatter_agent.run_as_node(state)
        
        assert "error_messages" in result
        assert "No CV data found in state" in result["error_messages"][0]
        assert "final_output_path" not in result

    @pytest.mark.asyncio
    @patch('src.agents.formatter_agent.logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.Environment')
    async def test_run_as_node_template_error(self, mock_env, mock_get_config, mock_logger,
                                       formatter_agent, sample_state):
        """Test handling of template rendering errors."""
        # Setup mocks
        mock_config = Mock()
        mock_config.project_root = Path("/test/project")
        mock_get_config.return_value = mock_config
        
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
    @patch('src.agents.formatter_agent.logger')
    @patch('src.agents.formatter_agent.get_config')
    @patch('src.agents.formatter_agent.Environment')
    async def test_run_as_node_css_file_missing(self, mock_env, mock_get_config, mock_logger,
                                          formatter_agent, sample_state):
        """Test handling when CSS file is missing."""
        # Setup mocks
        mock_config = Mock()
        mock_config.project_root = Path("/test/project")
        mock_get_config.return_value = mock_config
        
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test CV</html>"
        mock_jinja_env = Mock()
        mock_jinja_env.get_template.return_value = mock_template
        mock_env.return_value = mock_jinja_env
        
        mock_html_instance = Mock()
        mock_html_instance.write_pdf.return_value = b"fake pdf content"
        mock_html.return_value = mock_html_instance
        
        # Create temporary directory without CSS file
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config.project_root = Path(temp_dir)
            
            # Execute
            result = await formatter_agent.run_as_node(sample_state)
            
            # Verify it still works without CSS
            assert "final_output_path" in result
            # Verify PDF generation was called with no stylesheets
            mock_html_instance.write_pdf.assert_called_once_with(stylesheets=None)

    @patch('src.agents.formatter_agent.logger')
    def test_legacy_run_method(self, formatter_agent, mock_logger):
        """Test the legacy run method for backward compatibility."""
        # Setup mock LLM service directly on the agent instance
        mock_llm = Mock()
        mock_llm.generate_response.return_value = """
# Tailored CV

## Professional Profile

Test summary

---

## Key Qualifications

Python, Machine Learning

---

## Professional Experience

• Developed applications using Python and machine learning frameworks
• Led cross-functional team of 5 engineers to deliver project on time

---
""".strip()
        formatter_agent.llm_service = mock_llm
        
        input_data = {
            "content_data": {
                "summary": "Test summary",
                "skills_section": "Python, Machine Learning",
                "experience_bullets": [
                    "Developed applications using Python and machine learning frameworks",
                    "Led cross-functional team of 5 engineers to deliver project on time"
                ]
            },
            "format_specs": {}
        }
        
        result = formatter_agent.run(input_data)
        
        assert "formatted_cv_text" in result
        assert "# Tailored CV" in result["formatted_cv_text"]
        assert "Test summary" in result["formatted_cv_text"]
        assert "Python, Machine Learning" in result["formatted_cv_text"]

    def test_legacy_run_method_no_content(self, formatter_agent):
        """Test legacy run method with no content data."""
        input_data = {}
        
        result = formatter_agent.run(input_data)
        
        assert "formatted_cv_text" in result
        assert "error" in result
        assert "Missing content data" in result["error"]