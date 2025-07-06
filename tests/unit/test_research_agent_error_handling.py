"""Test for ResearchAgent error handling improvements."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from src.agents.research_agent import ResearchAgent
from src.models.agent_output_models import ResearchFindings, ResearchStatus, ResearchAgentOutput
from src.models.cv_models import JobDescriptionData
from src.models.agent_models import AgentResult
from src.error_handling.exceptions import LLMResponseParsingError, AgentExecutionError


class TestResearchAgentErrorHandling:
    """Test cases for ResearchAgent error handling improvements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_vector_store_service = Mock()
        self.mock_settings = {}
        self.mock_template_manager = Mock()
        self.mock_template_manager.templates = {}
        
        self.research_agent = ResearchAgent(
            llm_service=self.mock_llm_service,
            vector_store_service=self.mock_vector_store_service,
            settings=self.mock_settings,
            template_manager=self.mock_template_manager,
            session_id="test-session"
        )
        
        # Create valid job description data for testing
        self.valid_job_desc = JobDescriptionData(
            raw_text="Test job description",
            job_title="Software Engineer",
            company_name="Test Company",
            main_job_description_raw="Test description",
            skills=["Python", "Django"],
            experience_level="Mid-level",
            responsibilities=["Develop software"],
            industry_terms=["Agile"],
            company_values=["Innovation"]
        )

    def test_parse_llm_response_empty_response(self):
        """Test _parse_llm_response with empty LLM response."""
        with pytest.raises(LLMResponseParsingError, match="Empty or None LLM response received"):
            self.research_agent._parse_llm_response("")
        
        with pytest.raises(LLMResponseParsingError, match="Empty or None LLM response received"):
            self.research_agent._parse_llm_response(None)

    def test_parse_llm_response_invalid_json(self):
        """Test _parse_llm_response with invalid JSON."""
        invalid_json = "This is not valid JSON at all"
        
        with pytest.raises(LLMResponseParsingError, match="Failed to parse LLM response as JSON"):
            self.research_agent._parse_llm_response(invalid_json)

    def test_parse_llm_response_malformed_json(self):
        """Test _parse_llm_response with malformed JSON."""
        malformed_json = '{"key": "value", "incomplete": '
        
        with pytest.raises(LLMResponseParsingError, match="Failed to parse LLM response as JSON"):
            self.research_agent._parse_llm_response(malformed_json)

    def test_parse_llm_response_missing_required_fields(self):
        """Test _parse_llm_response with JSON missing required fields."""
        incomplete_json = json.dumps({
            "core_technical_skills": ["Python"]
            # Missing other required fields
        })
        
        with pytest.raises(LLMResponseParsingError, match="Failed to extract meaningful data from LLM response"):
            self.research_agent._parse_llm_response(incomplete_json)

    def test_parse_llm_response_valid_json(self):
        """Test _parse_llm_response with valid JSON response."""
        valid_json = json.dumps({
            "core_technical_skills": ["Python", "Django", "REST APIs"],
            "soft_skills": ["Communication", "Teamwork"],
            "key_performance_metrics": ["Code quality", "Delivery time"],
            "project_types": ["Web applications", "API development"],
            "working_environment_characteristics": ["Agile", "Remote-friendly"],
            "company_culture_indicators": ["Innovation", "Collaboration"],
            "industry_specific_requirements": ["GDPR compliance", "Security"]
        })
        
        result = self.research_agent._parse_llm_response(valid_json)
        
        assert isinstance(result, ResearchFindings)
        assert result.status == ResearchStatus.SUCCESS
        assert result.role_insights is not None
        assert "Python" in result.role_insights.required_skills
        assert "Communication" in result.role_insights.preferred_qualifications

    def test_parse_llm_response_fallback_to_text_extraction(self):
        """Test _parse_llm_response fallback to text extraction."""
        # Text that will trigger fallback but has extractable skills
        text_response = """
        The role requires strong Python and JavaScript skills.
        Communication and leadership are essential soft skills.
        Experience with Docker and Kubernetes is preferred.
        """
        
        result = self.research_agent._parse_llm_response(text_response)
        
        assert isinstance(result, ResearchFindings)
        assert result.status == ResearchStatus.PARTIAL
        assert result.role_insights is not None
        # Should extract some skills from text
        assert len(result.role_insights.required_skills) > 0

    @pytest.mark.asyncio
    async def test_perform_research_analysis_parsing_error(self):
        """Test _perform_research_analysis when parsing fails."""
        # Mock LLM service to return invalid response
        mock_response = Mock()
        mock_response.content = "Invalid response"
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_response)
        
        # Mock _parse_llm_response to raise LLMResponseParsingError
        with patch.object(self.research_agent, '_parse_llm_response', side_effect=LLMResponseParsingError("Parsing failed")):
            result = await self.research_agent._perform_research_analysis(self.valid_job_desc)
        
        assert isinstance(result, ResearchFindings)
        assert result.status == ResearchStatus.FAILED
        assert "Parsing failed" in result.error_message

    @pytest.mark.asyncio
    async def test_perform_research_analysis_none_result(self):
        """Test _perform_research_analysis when parsing returns None."""
        # Mock LLM service
        mock_response = Mock()
        mock_response.content = "Some response"
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_response)
        
        # Mock _parse_llm_response to return None
        with patch.object(self.research_agent, '_parse_llm_response', return_value=None):
            result = await self.research_agent._perform_research_analysis(self.valid_job_desc)
        
        assert isinstance(result, ResearchFindings)
        assert result.status == ResearchStatus.FAILED
        assert "Failed to parse LLM response - no findings extracted" in result.error_message

    @pytest.mark.asyncio
    async def test_perform_research_analysis_invalid_type(self):
        """Test _perform_research_analysis when parsing returns wrong type."""
        # Mock LLM service
        mock_response = Mock()
        mock_response.content = "Some response"
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_response)
        
        # Mock _parse_llm_response to return wrong type (string instead of ResearchFindings)
        with patch.object(self.research_agent, '_parse_llm_response', return_value="wrong_type"):
            result = await self.research_agent._perform_research_analysis(self.valid_job_desc)
        
        assert isinstance(result, ResearchFindings)
        assert result.status == ResearchStatus.FAILED
        assert "Invalid parsing result type: <class 'str'>" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_missing_job_description(self):
        """Test _execute with missing job description data."""
        result = await self.research_agent._execute(job_description_data=None)
        
        assert isinstance(result, AgentResult)
        assert not result.success
        assert "Missing required job description data" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_parsing_error_handling(self):
        """Test _execute handles LLMResponseParsingError correctly."""
        # Mock _perform_research_analysis to raise LLMResponseParsingError
        with patch.object(self.research_agent, '_perform_research_analysis', side_effect=LLMResponseParsingError("Parse error")):
            result = await self.research_agent._execute(job_description_data=self.valid_job_desc)
        
        assert isinstance(result, AgentResult)
        assert not result.success
        assert "Failed to parse LLM response: Parse error" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_agent_execution_error_handling(self):
        """Test _execute handles AgentExecutionError correctly."""
        # Mock _perform_research_analysis to raise AgentExecutionError
        with patch.object(self.research_agent, '_perform_research_analysis', side_effect=AgentExecutionError("TestAgent", "Execution error")):
            result = await self.research_agent._execute(job_description_data=self.valid_job_desc)
        
        assert isinstance(result, AgentResult)
        assert not result.success
        assert "Execution error" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_unexpected_error_handling(self):
        """Test _execute handles unexpected errors correctly."""
        # Mock _perform_research_analysis to raise unexpected error
        with patch.object(self.research_agent, '_perform_research_analysis', side_effect=ValueError("Unexpected error")):
            result = await self.research_agent._execute(job_description_data=self.valid_job_desc)
        
        assert isinstance(result, AgentResult)
        assert not result.success
        assert "An unexpected error occurred during research: Unexpected error" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_failed_research_status(self):
        """Test _execute when research analysis returns FAILED status."""
        # Create a failed research findings
        failed_findings = ResearchFindings(
            status=ResearchStatus.FAILED,
            error_message="Research failed for some reason"
        )
        
        # Mock _perform_research_analysis to return failed findings
        with patch.object(self.research_agent, '_perform_research_analysis', return_value=failed_findings):
            result = await self.research_agent._execute(job_description_data=self.valid_job_desc)
        
        assert isinstance(result, AgentResult)
        assert not result.success
        assert "Research failed for some reason" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_successful_research(self):
        """Test _execute with successful research analysis."""
        print("Starting test_execute_successful_research")
        
        # Create successful research findings
        try:
            successful_findings = ResearchFindings(
                status=ResearchStatus.SUCCESS
            )
            print(f"Created successful_findings: {successful_findings}")
            print(f"Status value: {successful_findings.status}")
            print(f"Status type: {type(successful_findings.status)}")
            
            # Test ResearchAgentOutput creation directly
            print("Testing ResearchAgentOutput creation...")
            test_output = ResearchAgentOutput(research_findings=successful_findings)
            print(f"ResearchAgentOutput created successfully: {test_output}")
            
        except Exception as e:
            print(f"Error creating ResearchFindings or ResearchAgentOutput: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Mock _perform_research_analysis to return successful findings
        print("Setting up mock")
        with patch.object(self.research_agent, '_perform_research_analysis', return_value=successful_findings):
            try:
                print("Calling _execute")
                result = await self.research_agent._execute(job_description_data=self.valid_job_desc)
                print(f"Result type: {type(result)}")
                print(f"Result success: {result.success}")
                print(f"Result metadata: {result.metadata}")
                print(f"Result error_message: {result.error_message}")
            except Exception as e:
                print(f"Exception during execution: {e}")
                print(f"Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        print("Running assertions")
        assert isinstance(result, AgentResult)
        assert result.success
        assert "Research analysis completed successfully" in result.metadata.get("message", "")