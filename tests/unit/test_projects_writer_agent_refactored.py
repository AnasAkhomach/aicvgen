"""Unit tests for the refactored ProjectsWriterAgent following Gold Standard LCEL pattern."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.projects_writer_agent import ProjectsWriterAgent, ProjectsWriterAgentInput
from src.models.agent_output_models import ProjectLLMOutput


class TestProjectsWriterAgentRefactored:
    """Test suite for the refactored ProjectsWriterAgent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock language model."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        return llm
    
    @pytest.fixture
    def mock_prompt(self):
        """Create a mock prompt template."""
        prompt = MagicMock()
        return prompt
    
    @pytest.fixture
    def mock_parser(self):
        """Create a mock output parser."""
        parser = MagicMock()
        parser.get_format_instructions.return_value = "Return a JSON object with project_description, technologies_used, achievements, and bullet_points."
        return parser
    
    @pytest.fixture
    def agent(self, mock_llm, mock_prompt, mock_parser):
        """Create a ProjectsWriterAgent instance."""
        return ProjectsWriterAgent(
            llm=mock_llm,
            prompt=mock_prompt,
            parser=mock_parser,
            settings={"temperature": 0.7},
            session_id="test_session"
        )
    
    @pytest.fixture
    def sample_input_data(self):
        """Create sample input data for testing."""
        return {
            "job_description": "We are looking for a Python developer with experience in web applications.",
            "project_item": {
                "id": "proj_1",
                "title": "E-commerce Platform",
                "description": "Built a full-stack e-commerce platform",
                "technologies": ["Python", "Django", "PostgreSQL"]
            },
            "key_qualifications": "Python development\nWeb application development\nDatabase design",
            "professional_experience": "5 years of Python development\nExperience with Django framework",
            "research_findings": {"company_focus": "E-commerce solutions"},
            "template_content": "Generate project content for: {project_item}",
            "format_instructions": "Return a JSON object with project_description, technologies_used, achievements, and bullet_points."
        }
    
    def test_agent_initialization(self, agent, mock_llm, mock_prompt, mock_parser):
        """Test that the agent initializes correctly with LCEL components."""
        assert agent.llm == mock_llm
        assert agent.prompt == mock_prompt
        assert agent.parser == mock_parser
        assert agent.settings == {"temperature": 0.7}
        assert agent.session_id == "test_session"
        assert agent.chain is not None
    
    def test_input_model_validation(self, sample_input_data):
        """Test that the ProjectsWriterAgentInput model validates correctly."""
        # Test valid input
        input_model = ProjectsWriterAgentInput(**sample_input_data)
        assert input_model.job_description == sample_input_data["job_description"]
        assert input_model.project_item == sample_input_data["project_item"]
        assert input_model.key_qualifications == sample_input_data["key_qualifications"]
        assert input_model.professional_experience == sample_input_data["professional_experience"]
        assert input_model.research_findings == sample_input_data["research_findings"]
        assert input_model.template_content == sample_input_data["template_content"]
        assert input_model.format_instructions == sample_input_data["format_instructions"]
    
    def test_input_model_validation_missing_required_field(self, sample_input_data):
        """Test that the input model raises validation error for missing required fields."""
        # Remove a required field
        del sample_input_data["job_description"]
        
        with pytest.raises(ValueError):
            ProjectsWriterAgentInput(**sample_input_data)
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent, sample_input_data):
        """Test successful execution of the agent."""
        # Mock the chain output
        expected_output = ProjectLLMOutput(
            project_description="Enhanced e-commerce platform with advanced features",
            technologies_used=["Python", "Django", "PostgreSQL", "Redis"],
            achievements=["Improved performance by 40%", "Increased user engagement"],
            bullet_points=[
                "Developed scalable e-commerce platform using Python and Django",
                "Implemented advanced caching with Redis for 40% performance improvement",
                "Designed robust PostgreSQL database schema for product management"
            ]
        )
        
        # Mock the chain.ainvoke method
        agent.chain.ainvoke = AsyncMock(return_value=expected_output)
        
        # Execute the agent
        result = await agent._execute(**sample_input_data)
        
        # Verify the result
        assert "generated_projects" in result
        assert result["generated_projects"] == expected_output
        
        # Verify the chain was called with the correct input
        agent.chain.ainvoke.assert_called_once()
        call_args = agent.chain.ainvoke.call_args[0][0]
        assert call_args["job_description"] == sample_input_data["job_description"]
        assert call_args["project_item"] == sample_input_data["project_item"]
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_input(self, agent):
        """Test that the agent raises validation error for invalid input."""
        invalid_input = {
            "job_description": "Test job description"
            # Missing required fields
        }
        
        with pytest.raises(ValueError):
            await agent._execute(**invalid_input)
    
    @pytest.mark.asyncio
    async def test_execute_chain_error(self, agent, sample_input_data):
        """Test that the agent properly handles chain execution errors."""
        # Mock the chain to raise an exception
        agent.chain.ainvoke = AsyncMock(side_effect=Exception("Chain execution failed"))
        
        with pytest.raises(Exception, match="Chain execution failed"):
            await agent._execute(**sample_input_data)
    
    def test_input_model_optional_fields(self, sample_input_data):
        """Test that optional fields work correctly in the input model."""
        # Remove optional field
        del sample_input_data["research_findings"]
        
        input_model = ProjectsWriterAgentInput(**sample_input_data)
        assert input_model.research_findings is None
    
    def test_input_model_serialization(self, sample_input_data):
        """Test that the input model can be serialized and deserialized."""
        input_model = ProjectsWriterAgentInput(**sample_input_data)
        
        # Test model_dump
        dumped_data = input_model.model_dump()
        assert isinstance(dumped_data, dict)
        assert dumped_data["job_description"] == sample_input_data["job_description"]
        
        # Test recreation from dumped data
        recreated_model = ProjectsWriterAgentInput(**dumped_data)
        assert recreated_model.job_description == input_model.job_description
        assert recreated_model.project_item == input_model.project_item