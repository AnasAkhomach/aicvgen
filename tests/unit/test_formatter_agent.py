#!/usr/bin/env python3
"""
Unit tests for FormatterAgent.

Tests the CV formatting functionality including PDF generation,
HTML formatting, and the run_as_node method for LangGraph integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, mock_open
from typing import Dict, Any
from pathlib import Path

from src.agents.formatter_agent import FormatterAgent
from src.agents.agent_base import AgentExecutionContext, AgentResult
from src.models.data_models import StructuredCV, Section, Item, ItemStatus
from src.models.workflow_models import AgentState


class TestFormatterAgent:
    """Test cases for FormatterAgent."""

    @pytest.fixture
    def formatter_agent(self):
        """Create a FormatterAgent instance for testing."""
        return FormatterAgent(
            name="TestFormatter",
            description="Test formatter agent"
        )

    @pytest.fixture
    def execution_context(self):
        """Create a test execution context."""
        return AgentExecutionContext(
            session_id="test_session_123",
            item_id="test_item",
            metadata={"test": "data"}
        )

    @pytest.fixture
    def sample_structured_cv(self):
        """Create a sample structured CV for testing."""
        cv = StructuredCV()
        cv.sections = [
            Section(
                name="Personal Information",
                items=[
                    Item(content="John Doe", status=ItemStatus.GENERATED),
                    Item(content="john.doe@email.com", status=ItemStatus.GENERATED),
                    Item(content="+1-555-0123", status=ItemStatus.GENERATED)
                ]
            ),
            Section(
                name="Professional Experience",
                items=[
                    Item(content="Senior Software Engineer at TechCorp (2020-2024)", status=ItemStatus.GENERATED),
                    Item(content="Led development of scalable web applications using Python and React", status=ItemStatus.GENERATED),
                    Item(content="Managed team of 5 developers and improved deployment efficiency by 40%", status=ItemStatus.GENERATED)
                ]
            ),
            Section(
                name="Technical Skills",
                items=[
                    Item(content="Programming Languages: Python, JavaScript, TypeScript, Java", status=ItemStatus.GENERATED),
                    Item(content="Frameworks: React, Django, Flask, Spring Boot", status=ItemStatus.GENERATED),
                    Item(content="Cloud Platforms: AWS, Azure, Docker, Kubernetes", status=ItemStatus.GENERATED)
                ]
            ),
            Section(
                name="Education",
                items=[
                    Item(content="Master of Science in Computer Science, MIT (2018-2020)", status=ItemStatus.GENERATED),
                    Item(content="Bachelor of Science in Software Engineering, Stanford (2014-2018)", status=ItemStatus.GENERATED)
                ]
            )
        ]
        return cv

    @pytest.fixture
    def sample_content_data(self, sample_structured_cv):
        """Create sample content data for testing."""
        return {
            "structured_cv": sample_structured_cv,
            "format_type": "pdf",
            "template_name": "professional",
            "output_path": "test_output.pdf"
        }

    @pytest.fixture
    def sample_agent_state(self, sample_structured_cv):
        """Create a sample agent state for testing."""
        return AgentState(
            job_description="Senior Software Engineer position",
            structured_cv=sample_structured_cv,
            content_data={
                "format_type": "pdf",
                "template_name": "professional",
                "output_path": "test_output.pdf"
            },
            error_messages=[]
        )

    def test_formatter_agent_initialization(self):
        """Test FormatterAgent initialization."""
        agent = FormatterAgent(
            name="TestFormatter",
            description="Test description"
        )
        
        assert agent.name == "TestFormatter"
        assert agent.description == "Test description"
        assert hasattr(agent, 'input_schema')
        assert hasattr(agent, 'output_schema')

    async def test_run_success(self, formatter_agent, sample_content_data):
        """Test successful formatting execution."""
        # Mock the formatting methods
        with patch.object(formatter_agent, '_generate_pdf') as mock_pdf, \
             patch.object(formatter_agent, '_validate_output') as mock_validate:
            
            mock_pdf.return_value = "test_output.pdf"
            mock_validate.return_value = True
            
            result = formatter_agent.run(sample_content_data)
            
            assert "final_output_path" in result
            assert result["final_output_path"] == "test_output.pdf"
            assert "format_type" in result
            assert result["format_type"] == "pdf"
            mock_pdf.assert_called_once()
            mock_validate.assert_called_once_with("test_output.pdf")

    async def test_run_missing_structured_cv(self, formatter_agent):
        """Test formatting execution with missing structured CV."""
        content_data = {
            "format_type": "pdf",
            "template_name": "professional"
        }
        
        with pytest.raises(ValueError, match="Missing required structured_cv"):
            formatter_agent.run(content_data)

    async def test_run_invalid_format_type(self, formatter_agent, sample_structured_cv):
        """Test formatting execution with invalid format type."""
        content_data = {
            "structured_cv": sample_structured_cv,
            "format_type": "invalid_format",
            "template_name": "professional"
        }
        
        with pytest.raises(ValueError, match="Unsupported format type"):
            formatter_agent.run(content_data)

    async def test_run_as_node_success(self, formatter_agent, sample_agent_state):
        """Test successful run_as_node execution."""
        # Mock the run method
        with patch.object(formatter_agent, 'run') as mock_run:
            mock_run.return_value = {
                "final_output_path": "test_output.pdf",
                "format_type": "pdf",
                "template_used": "professional"
            }
            
            result_state = await formatter_agent.run_as_node(sample_agent_state)
            
            assert result_state.final_output_path == "test_output.pdf"
            assert len(result_state.error_messages) == 0
            mock_run.assert_called_once()

    async def test_run_as_node_missing_content_data(self, formatter_agent, sample_structured_cv):
        """Test run_as_node with missing content data."""
        state = AgentState(
            job_description="Test job",
            structured_cv=sample_structured_cv,
            content_data=None,
            error_messages=[]
        )
        
        result_state = await formatter_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "Missing content_data" in result_state.error_messages[0]
        assert result_state.final_output_path is None

    async def test_run_as_node_missing_structured_cv(self, formatter_agent):
        """Test run_as_node with missing structured CV."""
        state = AgentState(
            job_description="Test job",
            structured_cv=None,
            content_data={"format_type": "pdf"},
            error_messages=[]
        )
        
        result_state = await formatter_agent.run_as_node(state)
        
        assert len(result_state.error_messages) == 1
        assert "Missing structured_cv" in result_state.error_messages[0]
        assert result_state.final_output_path is None

    async def test_run_as_node_formatting_error(self, formatter_agent, sample_agent_state):
        """Test run_as_node with formatting error."""
        # Mock the run method to raise an exception
        with patch.object(formatter_agent, 'run') as mock_run:
            mock_run.side_effect = Exception("PDF generation failed")
            
            result_state = await formatter_agent.run_as_node(sample_agent_state)
            
            assert len(result_state.error_messages) == 1
            assert "Error during formatting: PDF generation failed" in result_state.error_messages[0]
            assert result_state.final_output_path is None

    @patch('src.agents.formatter_agent.weasyprint')
    def test_generate_pdf_success(self, mock_weasyprint, formatter_agent, sample_structured_cv):
        """Test successful PDF generation."""
        # Mock weasyprint
        mock_html = Mock()
        mock_weasyprint.HTML.return_value = mock_html
        mock_html.write_pdf = Mock()
        
        # Mock HTML generation
        with patch.object(formatter_agent, '_generate_html') as mock_html_gen:
            mock_html_gen.return_value = "<html><body>Test CV</body></html>"
            
            output_path = formatter_agent._generate_pdf(
                sample_structured_cv,
                "professional",
                "test_output.pdf"
            )
            
            assert output_path == "test_output.pdf"
            mock_html_gen.assert_called_once_with(sample_structured_cv, "professional")
            mock_weasyprint.HTML.assert_called_once()
            mock_html.write_pdf.assert_called_once_with("test_output.pdf")

    @patch('src.agents.formatter_agent.weasyprint')
    def test_generate_pdf_weasyprint_error(self, mock_weasyprint, formatter_agent, sample_structured_cv):
        """Test PDF generation with weasyprint error."""
        # Mock weasyprint to raise an exception
        mock_weasyprint.HTML.side_effect = Exception("Weasyprint error")
        
        with patch.object(formatter_agent, '_generate_html') as mock_html_gen:
            mock_html_gen.return_value = "<html><body>Test CV</body></html>"
            
            with pytest.raises(Exception, match="Weasyprint error"):
                formatter_agent._generate_pdf(
                    sample_structured_cv,
                    "professional",
                    "test_output.pdf"
                )

    def test_generate_html_professional_template(self, formatter_agent, sample_structured_cv):
        """Test HTML generation with professional template."""
        html_content = formatter_agent._generate_html(sample_structured_cv, "professional")
        
        assert isinstance(html_content, str)
        assert "<html>" in html_content
        assert "<body>" in html_content
        assert "John Doe" in html_content
        assert "Senior Software Engineer" in html_content
        assert "Python" in html_content
        assert "MIT" in html_content

    def test_generate_html_modern_template(self, formatter_agent, sample_structured_cv):
        """Test HTML generation with modern template."""
        html_content = formatter_agent._generate_html(sample_structured_cv, "modern")
        
        assert isinstance(html_content, str)
        assert "<html>" in html_content
        assert "<body>" in html_content
        assert "John Doe" in html_content
        # Modern template should have different styling
        assert "class=" in html_content or "style=" in html_content

    def test_generate_html_creative_template(self, formatter_agent, sample_structured_cv):
        """Test HTML generation with creative template."""
        html_content = formatter_agent._generate_html(sample_structured_cv, "creative")
        
        assert isinstance(html_content, str)
        assert "<html>" in html_content
        assert "<body>" in html_content
        assert "John Doe" in html_content

    def test_generate_html_invalid_template(self, formatter_agent, sample_structured_cv):
        """Test HTML generation with invalid template."""
        # Should fall back to professional template
        html_content = formatter_agent._generate_html(sample_structured_cv, "invalid_template")
        
        assert isinstance(html_content, str)
        assert "<html>" in html_content
        assert "John Doe" in html_content

    def test_generate_html_empty_cv(self, formatter_agent):
        """Test HTML generation with empty CV."""
        empty_cv = StructuredCV()
        
        html_content = formatter_agent._generate_html(empty_cv, "professional")
        
        assert isinstance(html_content, str)
        assert "<html>" in html_content
        assert "<body>" in html_content
        # Should handle empty CV gracefully

    @patch('builtins.open', new_callable=mock_open)
    def test_generate_html_file_success(self, mock_file, formatter_agent, sample_structured_cv):
        """Test HTML file generation."""
        output_path = formatter_agent._generate_html_file(
            sample_structured_cv,
            "professional",
            "test_output.html"
        )
        
        assert output_path == "test_output.html"
        mock_file.assert_called_once_with("test_output.html", 'w', encoding='utf-8')
        # Verify that write was called
        mock_file().write.assert_called()

    @patch('builtins.open')
    def test_generate_html_file_write_error(self, mock_open_func, formatter_agent, sample_structured_cv):
        """Test HTML file generation with write error."""
        # Mock open to raise an exception
        mock_open_func.side_effect = IOError("Permission denied")
        
        with pytest.raises(IOError, match="Permission denied"):
            formatter_agent._generate_html_file(
                sample_structured_cv,
                "professional",
                "test_output.html"
            )

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_validate_output_success(self, mock_stat, mock_exists, formatter_agent):
        """Test successful output validation."""
        mock_exists.return_value = True
        mock_stat.return_value.st_size = 1024  # 1KB file
        
        is_valid = formatter_agent._validate_output("test_output.pdf")
        
        assert is_valid is True
        mock_exists.assert_called_once()
        mock_stat.assert_called_once()

    @patch('pathlib.Path.exists')
    def test_validate_output_file_not_exists(self, mock_exists, formatter_agent):
        """Test output validation with non-existent file."""
        mock_exists.return_value = False
        
        is_valid = formatter_agent._validate_output("nonexistent.pdf")
        
        assert is_valid is False
        mock_exists.assert_called_once()

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_validate_output_empty_file(self, mock_stat, mock_exists, formatter_agent):
        """Test output validation with empty file."""
        mock_exists.return_value = True
        mock_stat.return_value.st_size = 0  # Empty file
        
        is_valid = formatter_agent._validate_output("empty.pdf")
        
        assert is_valid is False
        mock_exists.assert_called_once()
        mock_stat.assert_called_once()

    def test_format_section_content(self, formatter_agent):
        """Test section content formatting."""
        section = Section(
            name="Professional Experience",
            items=[
                Item(content="Senior Developer at ABC Corp", status=ItemStatus.GENERATED),
                Item(content="Led team of 5 developers", status=ItemStatus.GENERATED)
            ]
        )
        
        formatted_content = formatter_agent._format_section_content(section)
        
        assert isinstance(formatted_content, str)
        assert "Senior Developer at ABC Corp" in formatted_content
        assert "Led team of 5 developers" in formatted_content

    def test_format_section_content_empty_section(self, formatter_agent):
        """Test section content formatting with empty section."""
        section = Section(name="Empty Section", items=[])
        
        formatted_content = formatter_agent._format_section_content(section)
        
        assert isinstance(formatted_content, str)
        assert len(formatted_content.strip()) == 0 or "No content" in formatted_content

    def test_apply_template_styling_professional(self, formatter_agent):
        """Test template styling application for professional template."""
        content = "<div>Test content</div>"
        
        styled_content = formatter_agent._apply_template_styling(content, "professional")
        
        assert isinstance(styled_content, str)
        assert "<style>" in styled_content or "style=" in styled_content
        assert content in styled_content

    def test_apply_template_styling_modern(self, formatter_agent):
        """Test template styling application for modern template."""
        content = "<div>Test content</div>"
        
        styled_content = formatter_agent._apply_template_styling(content, "modern")
        
        assert isinstance(styled_content, str)
        assert content in styled_content

    def test_apply_template_styling_creative(self, formatter_agent):
        """Test template styling application for creative template."""
        content = "<div>Test content</div>"
        
        styled_content = formatter_agent._apply_template_styling(content, "creative")
        
        assert isinstance(styled_content, str)
        assert content in styled_content

    def test_get_output_filename(self, formatter_agent):
        """Test output filename generation."""
        # Test with provided filename
        filename = formatter_agent._get_output_filename("pdf", "custom_name.pdf")
        assert filename == "custom_name.pdf"
        
        # Test without provided filename
        filename = formatter_agent._get_output_filename("pdf", None)
        assert filename.endswith(".pdf")
        assert "cv_" in filename
        
        # Test HTML format
        filename = formatter_agent._get_output_filename("html", None)
        assert filename.endswith(".html")

    def test_confidence_score_calculation(self, formatter_agent):
        """Test confidence score calculation."""
        # Test with successful formatting
        successful_output = {
            "final_output_path": "test_output.pdf",
            "format_type": "pdf",
            "template_used": "professional",
            "file_size": 1024
        }
        
        confidence = formatter_agent.get_confidence_score(successful_output)
        assert confidence > 0.8  # Should be high for successful formatting
        
        # Test with minimal output
        minimal_output = {
            "final_output_path": "test.pdf"
        }
        
        confidence = formatter_agent.get_confidence_score(minimal_output)
        assert confidence < 0.7  # Should be lower for minimal output

    @patch('src.agents.formatter_agent.logger')
    async def test_run_as_node_logs_execution(self, mock_logger, formatter_agent, sample_agent_state):
        """Test that run_as_node logs execution properly."""
        with patch.object(formatter_agent, 'run') as mock_run:
            mock_run.return_value = {"final_output_path": "test.pdf"}
            
            await formatter_agent.run_as_node(sample_agent_state)
            
            # Verify logging calls
            mock_logger.info.assert_called()
            assert any("Starting CV formatting" in str(call) for call in mock_logger.info.call_args_list)

    def test_sanitize_filename(self, formatter_agent):
        """Test filename sanitization."""
        # Test with special characters
        sanitized = formatter_agent._sanitize_filename("test/file<name>.pdf")
        assert "/" not in sanitized
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert sanitized.endswith(".pdf")
        
        # Test with normal filename
        sanitized = formatter_agent._sanitize_filename("normal_file.pdf")
        assert sanitized == "normal_file.pdf"

    def test_get_template_css_professional(self, formatter_agent):
        """Test CSS retrieval for professional template."""
        css = formatter_agent._get_template_css("professional")
        
        assert isinstance(css, str)
        assert len(css) > 0
        assert "font-family" in css or "color" in css

    def test_get_template_css_invalid(self, formatter_agent):
        """Test CSS retrieval for invalid template."""
        css = formatter_agent._get_template_css("invalid_template")
        
        assert isinstance(css, str)
        # Should return default CSS
        assert len(css) > 0