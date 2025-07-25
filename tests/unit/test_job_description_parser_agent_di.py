"""Test for JobDescriptionParserAgent dependency injection refactoring."""

import pytest
from unittest.mock import Mock, AsyncMock

from src.agents.job_description_parser_agent import JobDescriptionParserAgent
from src.services.llm_cv_parser_service import LLMCVParserService
from src.services.llm_service_interface import LLMServiceInterface
from src.templates.content_templates import ContentTemplateManager
from src.models.data_models import JobDescriptionData
from src.error_handling.exceptions import AgentExecutionError


class TestJobDescriptionParserAgentDI:
    """Test class for JobDescriptionParserAgent dependency injection."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        return Mock(spec=LLMServiceInterface)

    @pytest.fixture
    def mock_template_manager(self):
        """Create a mock template manager."""
        return Mock(spec=ContentTemplateManager)

    @pytest.fixture
    def mock_llm_cv_parser_service(self):
        """Create a mock LLM CV parser service."""
        mock_service = Mock(spec=LLMCVParserService)
        mock_service.parse_job_description_with_llm = AsyncMock()
        return mock_service

    @pytest.fixture
    def agent_settings(self):
        """Create test agent settings."""
        return {
            "system_instruction": "Test system instruction",
            "max_retries": 3,
            "timeout": 30
        }

    @pytest.fixture
    def job_description_parser_agent(
        self, 
        mock_llm_service, 
        mock_llm_cv_parser_service, 
        mock_template_manager, 
        agent_settings
    ):
        """Create a JobDescriptionParserAgent instance with mocked dependencies."""
        return JobDescriptionParserAgent(
            llm_service=mock_llm_service,
            llm_cv_parser_service=mock_llm_cv_parser_service,
            template_manager=mock_template_manager,
            settings=agent_settings,
            session_id="test-session-123"
        )

    def test_agent_initialization_with_dependency_injection(
        self, 
        job_description_parser_agent, 
        mock_llm_cv_parser_service
    ):
        """Test that the agent is properly initialized with injected dependencies."""
        # Verify the agent was initialized correctly
        assert job_description_parser_agent.name == "JobDescriptionParserAgent"
        assert job_description_parser_agent.session_id == "test-session-123"
        
        # Verify the injected service is used (not instantiated internally)
        assert job_description_parser_agent.llm_cv_parser_service is mock_llm_cv_parser_service

    def test_agent_does_not_instantiate_service_internally(
        self, 
        job_description_parser_agent, 
        mock_llm_cv_parser_service
    ):
        """Test that the agent uses the injected service instead of creating its own."""
        # The agent should use the injected mock service
        assert job_description_parser_agent.llm_cv_parser_service is mock_llm_cv_parser_service
        
        # Verify it's the exact mock instance, not a new instance
        assert isinstance(job_description_parser_agent.llm_cv_parser_service, Mock)

    @pytest.mark.asyncio
    async def test_parse_job_description_uses_injected_service(
        self, 
        job_description_parser_agent, 
        mock_llm_cv_parser_service
    ):
        """Test that parse_job_description uses the injected service."""
        # Setup mock return value
        expected_job_data = JobDescriptionData(
            raw_text="Test job description",
            job_title="Software Engineer",
            company_name="Test Company"
        )
        mock_llm_cv_parser_service.parse_job_description_with_llm.return_value = expected_job_data
        
        # Call the method
        result = await job_description_parser_agent.parse_job_description("Test job description")
        
        # Verify the injected service was called
        mock_llm_cv_parser_service.parse_job_description_with_llm.assert_called_once_with(
            "Test job description",
            session_id="test-session-123",
            system_instruction="Test system instruction"
        )
        
        # Verify the result
        assert result == expected_job_data

    @pytest.mark.asyncio
    async def test_execute_method_integration(
        self, 
        job_description_parser_agent, 
        mock_llm_cv_parser_service
    ):
        """Test the _execute method works with dependency injection."""
        # Setup mock return value
        expected_job_data = JobDescriptionData(
            raw_text="Test job description",
            job_title="Software Engineer",
            company_name="Test Company"
        )
        mock_llm_cv_parser_service.parse_job_description_with_llm.return_value = expected_job_data
        
        # Call the execute method
        result = await job_description_parser_agent._execute(
            input_data={"raw_text": "Test job description"}
        )
        
        # Verify the result structure
        assert "job_description_data" in result
        assert result["job_description_data"] == expected_job_data
        
        # Verify the injected service was called
        mock_llm_cv_parser_service.parse_job_description_with_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_raw_text_handling(
        self, 
        job_description_parser_agent, 
        mock_llm_cv_parser_service
    ):
        """Test handling of empty raw text."""
        # Call with empty text
        result = await job_description_parser_agent.parse_job_description("")
        
        # Should return JobDescriptionData with just raw_text
        assert isinstance(result, JobDescriptionData)
        assert result.raw_text == ""
        
        # Service should not be called for empty text
        mock_llm_cv_parser_service.parse_job_description_with_llm.assert_not_called()

    def test_constructor_signature_compliance(self):
        """Test that the constructor signature matches the expected DI pattern."""
        import inspect
        
        # Get the constructor signature
        sig = inspect.signature(JobDescriptionParserAgent.__init__)
        params = list(sig.parameters.keys())
        
        # Verify the expected parameters are present
        expected_params = [
            'self', 
            'llm_service', 
            'llm_cv_parser_service', 
            'template_manager', 
            'settings', 
            'session_id'
        ]
        
        assert params == expected_params
        
        # Verify the llm_cv_parser_service parameter has the correct type annotation
        llm_cv_parser_param = sig.parameters['llm_cv_parser_service']
        assert llm_cv_parser_param.annotation == LLMCVParserService