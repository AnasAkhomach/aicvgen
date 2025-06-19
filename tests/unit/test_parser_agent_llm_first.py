"""Unit tests for ParserAgent LLM-first CV parsing functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.agents.parser_agent import ParserAgent
from src.models.data_models import (
    StructuredCV, 
    JobDescriptionData, 
    CVParsingResult,
    CVParsingPersonalInfo,
    CVParsingSection,
    CVParsingSubsection,
    Section,
    Subsection,
    Item,
    ItemType,
    ItemStatus
)
from src.services.llm_service import EnhancedLLMService


class TestParserAgentLLMFirst:
    """Test cases for LLM-first CV parsing functionality."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock_service = Mock(spec=EnhancedLLMService)
        mock_service.generate_content = AsyncMock()
        return mock_service

    @pytest.fixture
    def parser_agent(self, mock_llm_service):
        """Create a ParserAgent instance with mocked dependencies."""
        agent = ParserAgent(
            name="test_parser_agent",
            description="Test parser agent for unit testing",
            llm_service=mock_llm_service
        )
        return agent

    @pytest.fixture
    def sample_cv_text(self):
        """Sample CV text for testing."""
        return """
John Doe
Email: john.doe@email.com | Phone: +1234567890
LinkedIn: https://linkedin.com/in/johndoe | GitHub: https://github.com/johndoe
Location: San Francisco, CA

## Executive Summary
Experienced software engineer with 5+ years in full-stack development.

## Professional Experience

### Senior Software Engineer at TechCorp (2020-2023)
- Developed scalable web applications using React and Node.js
- Led a team of 3 junior developers
- Improved system performance by 40%

### Software Engineer at StartupXYZ (2018-2020)
- Built microservices architecture using Python and Docker
- Implemented CI/CD pipelines

## Education

### Bachelor of Science in Computer Science
University of Technology (2014-2018)
- GPA: 3.8/4.0
- Relevant coursework: Data Structures, Algorithms, Software Engineering

## Key Qualifications
- Python, JavaScript, React, Node.js
- AWS, Docker, Kubernetes
- Agile/Scrum methodologies
"""

    @pytest.fixture
    def sample_llm_response(self):
        """Sample LLM response for testing."""
        return {
            "personal_info": {
                "name": "John Doe",
                "email": "john.doe@email.com",
                "phone": "+1234567890",
                "linkedin": "https://linkedin.com/in/johndoe",
                "github": "https://github.com/johndoe",
                "location": "San Francisco, CA"
            },
            "sections": [
                {
                    "name": "Executive Summary",
                    "subsections": [],
                    "items": [
                        "Experienced software engineer with 5+ years in full-stack development."
                    ]
                },
                {
                    "name": "Professional Experience",
                    "subsections": [],
                    "items": [
                        "Senior Software Engineer | Tech Corp | 2020-2023 | San Francisco, CA"
                    ]
                },
                {
                    "name": "Education",
                    "subsections": [],
                    "items": [
                        "Bachelor of Science in Computer Science | University of Technology | 2016-2020"
                    ]
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_parse_cv_with_llm_success(self, parser_agent, sample_cv_text, sample_llm_response):
        """Test successful CV parsing with LLM."""
        # Mock the _generate_and_parse_json method to return our sample response
        with patch.object(parser_agent, '_generate_and_parse_json', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = sample_llm_response
            
            # Mock job data
            job_data = JobDescriptionData(raw_text="Software Engineer position")
            
            # Execute the parsing
            result = await parser_agent.parse_cv_with_llm(sample_cv_text, job_data)
            
            # Verify the result
            assert isinstance(result, StructuredCV)
            assert result.metadata["name"] == "John Doe"
            assert result.metadata["email"] == "john.doe@email.com"
            assert len(result.sections) == 3
            
            # Check Executive Summary section
            exec_summary = result.sections[0]
            assert exec_summary.name == "Executive Summary"
            assert exec_summary.content_type == "DYNAMIC"
            assert len(exec_summary.items) == 1
            assert "Experienced software engineer" in exec_summary.items[0].content
            
            # Check Professional Experience section
            prof_exp = result.sections[1]
            assert prof_exp.name == "Professional Experience"
            assert len(prof_exp.items) == 1
            assert "Senior Software Engineer" in prof_exp.items[0].content
            
            # Verify the generate method was called with correct parameters
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            
    @pytest.mark.asyncio
    async def test_parse_cv_with_llm_prompt_loading(self, parser_agent, sample_cv_text, sample_llm_response):
        """Test that CV parsing correctly loads prompt from settings."""
        # Mock the settings and prompt loading
        with patch.object(parser_agent.settings, 'get_prompt_path_by_key') as mock_get_path, \
             patch('builtins.open', create=True) as mock_open, \
             patch.object(parser_agent, '_generate_and_parse_json', new_callable=AsyncMock) as mock_generate:
            
            # Setup mocks
            mock_get_path.return_value = "/fake/path/cv_parsing_prompt.md"
            mock_file = Mock()
            mock_file.read.return_value = "You are an expert CV parser. Parse: {{raw_cv_text}}"
            mock_open.return_value.__enter__.return_value = mock_file
            mock_generate.return_value = sample_llm_response
            
            # Mock job data
            job_data = JobDescriptionData(raw_text="Software Engineer position")
            
            # Execute the parsing
            result = await parser_agent.parse_cv_with_llm(sample_cv_text, job_data)
            
            # Verify settings method was called correctly
            mock_get_path.assert_called_once_with("cv_parser")
            
            # Verify file was opened correctly
            mock_open.assert_called_once_with("/fake/path/cv_parsing_prompt.md", "r", encoding="utf-8")
            
            # Verify prompt template replacement worked
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[1] if mock_generate.call_args[1] else mock_generate.call_args[0]
            prompt_used = call_args['prompt'] if 'prompt' in call_args else call_args[0]
            assert sample_cv_text in prompt_used
            assert "{{raw_cv_text}}" not in prompt_used  # Template should be replaced

    @pytest.mark.asyncio
    async def test_parse_cv_with_llm_empty_input(self, parser_agent):
        """Test CV parsing with empty input."""
        job_data = JobDescriptionData(raw_text="")
        
        result = await parser_agent.parse_cv_with_llm("", job_data)
        
        # Should return an empty StructuredCV
        assert isinstance(result, StructuredCV)
        assert len(result.sections) == 0
        assert result.metadata.get("original_cv_text") == ""

    @pytest.mark.asyncio
    async def test_parse_cv_with_llm_llm_error(self, parser_agent, sample_cv_text):
        """Test CV parsing when LLM fails."""
        # Mock the _generate_and_parse_json method to raise an exception
        with patch.object(parser_agent, '_generate_and_parse_json', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("LLM service error")
            
            job_data = JobDescriptionData(raw_text="Software Engineer position")
            
            result = await parser_agent.parse_cv_with_llm(sample_cv_text, job_data)
            
            # Should return an empty StructuredCV with error metadata
            assert isinstance(result, StructuredCV)
            assert len(result.sections) == 0
            assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_convert_parsing_result_to_structured_cv(self, parser_agent, sample_llm_response):
        """Test conversion from LLM parsing result to StructuredCV."""
        # Create a CVParsingResult from the sample response
        parsing_result = CVParsingResult(**sample_llm_response)
        job_data = JobDescriptionData(raw_text="Software Engineer position")
        
        # Test the conversion method
        result = parser_agent._convert_parsing_result_to_structured_cv(
            parsing_result, "sample cv text", job_data
        )
        
        # Verify the conversion
        assert isinstance(result, StructuredCV)
        assert result.metadata["name"] == "John Doe"
        assert result.metadata["original_cv_text"] == "sample cv text"
        assert len(result.sections) == 3
        
        # Verify section conversion
        exec_summary = result.sections[0]
        assert exec_summary.name == "Executive Summary"
        assert exec_summary.content_type == "DYNAMIC"
        assert len(exec_summary.items) == 1
        
        # Verify subsection conversion
        prof_exp = result.sections[1]
        assert len(prof_exp.subsections) == 1
        subsection = prof_exp.subsections[0]
        assert subsection.name == "Senior Software Engineer at TechCorp (2020-2023)"
        assert len(subsection.items) == 2

    def test_parse_cv_text_backward_compatibility(self, parser_agent, sample_cv_text):
        """Test that the synchronous parse_cv_text method still works for backward compatibility."""
        job_data = JobDescriptionData(raw_text="Software Engineer position")
        
        # Mock the async method
        with patch.object(parser_agent, 'parse_cv_with_llm', new_callable=AsyncMock) as mock_async:
            expected_result = StructuredCV(sections=[])
            mock_async.return_value = expected_result
            
            # Call the synchronous wrapper
            result = parser_agent.parse_cv_text(sample_cv_text, job_data)
            
            # Verify it calls the async method and returns the result
            assert result == expected_result
            mock_async.assert_called_once_with(sample_cv_text, job_data)

    @pytest.mark.asyncio
    async def test_parse_cv_with_llm_malformed_response(self, parser_agent, sample_cv_text):
        """Test CV parsing when LLM returns malformed response."""
        # Mock the _generate_and_parse_json method to return malformed data
        malformed_response = {
            "personal_info": {"name": "John Doe"},  # Missing required fields
            "sections": "not_a_list"  # Wrong type
        }
        
        with patch.object(parser_agent, '_generate_and_parse_json', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = malformed_response
            
            job_data = JobDescriptionData(raw_text="Software Engineer position")
            
            result = await parser_agent.parse_cv_with_llm(sample_cv_text, job_data)
            
            # Should handle the error gracefully and return empty CV
            assert isinstance(result, StructuredCV)
            assert len(result.sections) == 0
            assert "error" in result.metadata