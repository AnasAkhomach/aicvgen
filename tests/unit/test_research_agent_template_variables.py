"""Test for ResearchAgent template fix - ensuring correct template usage."""

import pytest
from unittest.mock import Mock, patch
from src.agents.research_agent import ResearchAgent
from src.models.cv_models import JobDescriptionData
from src.templates.content_templates import ContentTemplate, TemplateCategory
from src.models.workflow_models import ContentType
from src.models.agent_output_models import ResearchStatus


class TestResearchAgentTemplateFix:
    """Test class for ResearchAgent template fix."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service."""
        mock_service = Mock()
        mock_response = Mock()
        mock_response.content = """{
            "core_technical_skills": ["Python", "Django", "REST APIs"],
            "soft_skills": ["Communication", "Teamwork"],
            "key_performance_metrics": ["Code quality", "Delivery time"],
            "project_types": ["Web applications", "API development"],
            "working_environment_characteristics": ["Agile", "Remote-friendly"]
        }"""
        mock_service.generate_content.return_value = mock_response
        return mock_service

    @pytest.fixture
    def mock_vector_store_service(self):
        """Mock vector store service."""
        return Mock()

    @pytest.fixture
    def mock_template_manager(self):
        """Mock template manager with the correct job_research_analysis template."""
        mock_manager = Mock()

        # Create the correct template
        job_research_template = ContentTemplate(
            name="job_research_analysis",
            category=TemplateCategory.PROMPT,
            content_type=ContentType.JOB_ANALYSIS,
            template="""Analyze the following job description and provide a structured analysis.
Extract the following information:

1. Core technical skills required (list the top 5-7 most important)
2. Soft skills that would be valuable (list the top 3-5)
3. Key performance metrics mentioned or implied
4. Project types the candidate would likely work on
5. Working environment characteristics (team size, collaboration style, etc.)

Format your response as a JSON object with these 5 keys.

Job Description:
{raw_jd}

Additional context:
- Skills already identified: {skills}
- Company: {company_name}
- Position: {job_title}

Return your analysis as a well-structured JSON object with the specified keys.""",
            variables=["raw_jd", "skills", "company_name", "job_title"],
            description="Analyzes a job description and provides a structured analysis.",
        )

        # Mock the templates dictionary
        mock_manager.templates = {"job_research_analysis": job_research_template}

        # Mock format_template method
        mock_manager.format_template.return_value = "Formatted research prompt"

        return mock_manager

    @pytest.fixture
    def sample_job_description_data(self):
        """Sample job description data."""
        return JobDescriptionData(
            raw_text="We are looking for a Senior Python Developer to join our team...",
            job_title="Senior Python Developer",
            company_name="Tech Corp",
            main_job_description_raw="We are looking for a Senior Python Developer to join our team...",
            skills=["Python", "Django", "PostgreSQL"],
            responsibilities=["Develop web applications", "Code reviews"],
        )

    @pytest.fixture
    def research_agent(
        self, mock_llm_service, mock_vector_store_service, mock_template_manager
    ):
        """Create ResearchAgent instance with mocked dependencies."""
        settings = {"max_tokens_analysis": 1000, "temperature_analysis": 0.7}

        return ResearchAgent(
            llm_service=mock_llm_service,
            vector_store_service=mock_vector_store_service,
            settings=settings,
            template_manager=mock_template_manager,
            session_id="test_session",
        )

    def test_create_research_prompt_uses_correct_template(
        self, research_agent, sample_job_description_data, mock_template_manager
    ):
        """Test that _create_research_prompt uses the correct template and variables."""
        # Call the method directly
        prompt = research_agent._create_research_prompt(sample_job_description_data)

        # Verify the correct template was accessed
        assert "job_research_analysis" in mock_template_manager.templates

        # Verify format_template was called with correct variables
        mock_template_manager.format_template.assert_called_once()
        call_args = mock_template_manager.format_template.call_args

        # Check that the template object is correct
        template_arg = call_args[0][0]
        assert template_arg.name == "job_research_analysis"

        # Check that the variables are correct
        variables_arg = call_args[0][1]
        expected_variables = {
            "raw_jd": "We are looking for a Senior Python Developer to join our team...",
            "skills": "Python, Django, PostgreSQL",
            "company_name": "Tech Corp",
            "job_title": "Senior Python Developer",
        }
        assert variables_arg == expected_variables

    def test_create_research_prompt_with_empty_skills(
        self, research_agent, mock_template_manager
    ):
        """Test that _create_research_prompt handles empty skills correctly."""
        # Create job data with no skills
        job_data = JobDescriptionData(
            raw_text="Job description",
            job_title="Developer",
            company_name="Company",
            main_job_description_raw="Job description",
            skills=[],  # Empty skills
            responsibilities=[],
        )

        # Call the method
        prompt = research_agent._create_research_prompt(job_data)

        # Verify skills variable is set to "Not specified"
        call_args = mock_template_manager.format_template.call_args
        variables_arg = call_args[0][1]
        assert variables_arg["skills"] == "Not specified"

    def test_create_research_prompt_handles_missing_template(
        self, mock_llm_service, mock_vector_store_service, sample_job_description_data
    ):
        """Test that _create_research_prompt handles missing template gracefully."""
        # Create template manager without the job_research_analysis template
        mock_template_manager = Mock()
        mock_template_manager.templates = {}  # Empty templates

        settings = {"max_tokens_analysis": 1000, "temperature_analysis": 0.7}

        research_agent = ResearchAgent(
            llm_service=mock_llm_service,
            vector_store_service=mock_vector_store_service,
            settings=settings,
            template_manager=mock_template_manager,
            session_id="test_session",
        )

        # Call the method - should use fallback prompt
        prompt = research_agent._create_research_prompt(sample_job_description_data)

        # Should return a fallback prompt containing job details
        assert "Senior Python Developer" in prompt
        assert "Tech Corp" in prompt
        assert "JSON object" in prompt

    def test_template_fix_no_keyerror(
        self, research_agent, sample_job_description_data, mock_template_manager
    ):
        """Test that the template fix prevents KeyError for missing 'skills' variable."""
        # This test specifically verifies that the fix prevents the original KeyError
        # by ensuring the correct variables are passed to the template

        # Call the method that was causing the KeyError
        try:
            prompt = research_agent._create_research_prompt(sample_job_description_data)
            # If we get here without exception, the fix worked
            assert prompt is not None
        except KeyError as e:
            pytest.fail(f"KeyError still occurs: {e}")

        # Verify the template was called with all required variables
        mock_template_manager.format_template.assert_called_once()
        call_args = mock_template_manager.format_template.call_args
        variables_arg = call_args[0][1]

        # Ensure all required template variables are present
        required_vars = ["raw_jd", "skills", "company_name", "job_title"]
        for var in required_vars:
            assert (
                var in variables_arg
            ), f"Required variable '{var}' missing from template call"
