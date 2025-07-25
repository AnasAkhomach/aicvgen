"""Unit tests for ProjectsUpdaterAgent."""

import pytest
from unittest.mock import Mock, patch
from src.agents.projects_updater_agent import ProjectsUpdaterAgent
from src.models.cv_models import StructuredCV, Section, Item
from src.models.agent_input_models import ProjectsUpdaterAgentInput
from src.models.agent_output_models import ProjectsUpdaterAgentOutput
from src.models.cv_models import ItemStatus

pytestmark = pytest.mark.asyncio


class TestProjectsUpdaterAgent:
    """Test cases for ProjectsUpdaterAgent."""

    @pytest.fixture
    def mock_structured_cv(self):
        """Create a mock structured CV with existing projects section."""
        return StructuredCV(
            sections=[
                Section(
                    name="Projects",
                    items=[
                        Item(
                            content="Old projects content",
                            status=ItemStatus.INITIAL
                        )
                    ]
                ),
                Section(
                    name="Education",
                    items=[]
                )
            ]
        )

    @pytest.fixture
    def generated_projects_data(self):
        """Create mock generated projects data."""
        return "E-commerce Analytics Dashboard\n• Built real-time analytics dashboard using Python and Streamlit\n\nCustomer Segmentation ML Model\n• Developed machine learning model for customer segmentation"

    @pytest.fixture
    def agent_input(self, mock_structured_cv, generated_projects_data):
        """Create agent input with structured CV and generated data."""
        return ProjectsUpdaterAgentInput(
            structured_cv=mock_structured_cv,
            generated_projects=generated_projects_data
        )

    @pytest.fixture
    def updater_agent(self):
        """Create ProjectsUpdaterAgent instance."""
        return ProjectsUpdaterAgent(
            session_id="test_session",
            name="test_projects_updater"
        )

    def test_agent_initialization(self, updater_agent):
        """Test agent initializes correctly."""
        assert updater_agent.name == "test_projects_updater"
        assert updater_agent.session_id == "test_session"

    def test_input_validation_success(self, updater_agent, agent_input):
        """Test successful input validation."""
        # Should not raise any exception
        updater_agent._validate_inputs(agent_input.model_dump())

    def test_input_validation_missing_structured_cv(self, updater_agent, generated_projects_data):
        """Test input validation fails when structured_cv is missing."""
        invalid_input = ProjectsUpdaterAgentInput(
            structured_cv=None,
            generated_projects=generated_projects_data
        )
        
        with pytest.raises(ValueError, match="structured_cv is required"):
            updater_agent._validate_inputs(invalid_input.model_dump())

    def test_input_validation_missing_generated_data(self, updater_agent, mock_structured_cv):
        """Test input validation fails when generated data is missing."""
        invalid_input = ProjectsUpdaterAgentInput(
            structured_cv=mock_structured_cv,
            generated_projects=None
        )
        
        with pytest.raises(ValueError, match="generated_projects is required"):
            updater_agent._validate_inputs(invalid_input.model_dump())

    def test_find_projects_section_exists(self, updater_agent, mock_structured_cv):
        """Test finding existing projects section."""
        section = updater_agent._find_projects_section(mock_structured_cv)
        assert section is not None
        assert section.title == "Projects"

    def test_find_projects_section_not_exists(self, updater_agent):
        """Test finding projects section when it doesn't exist."""
        cv_without_projects = StructuredCV(
            sections=[
                Section(title="Education", items=[])
            ]
        )
        
        section = updater_agent._find_projects_section(cv_without_projects)
        assert section is None

    def test_create_projects_section(self, updater_agent):
        """Test creating new projects section."""
        cv_without_projects = StructuredCV(sections=[])
        
        section = updater_agent._create_projects_section(cv_without_projects)
        
        assert section.title == "Projects"
        assert len(section.items) == 0
        assert len(cv_without_projects.sections) == 1
        assert cv_without_projects.sections[0] == section

    def test_update_section_with_generated_data(self, updater_agent, generated_projects_data):
        """Test updating section with generated projects data."""
        section = Section(title="Projects", items=[])
        
        updater_agent._update_section_with_generated_data(section, generated_projects_data)
        
        assert len(section.items) == 2
        
        # Check first item
        assert section.items[0].content == "E-commerce Analytics Dashboard\n• Built real-time analytics dashboard using Python and Streamlit"
        assert section.items[0].status == ItemStatus.GENERATED
        assert section.items[0].metadata == {"type": "web_application", "technologies": ["Python", "Streamlit"]}
        
        # Check second item
        assert section.items[1].content == "Customer Segmentation ML Model\n• Developed machine learning model for customer segmentation"
        assert section.items[1].status == ItemStatus.GENERATED
        assert section.items[1].metadata == {"type": "machine_learning", "technologies": ["Python", "scikit-learn"]}

    async def test_execute_with_existing_section(self, updater_agent, agent_input):
        """Test execute method with existing projects section."""
        result = await updater_agent._execute(**agent_input.model_dump())
        
        assert isinstance(result, dict)
        assert result["structured_cv"] is not None
        
        # Find the projects section
        projects_section = None
        for section in result["structured_cv"].sections:
            if section.name == "Projects":
                projects_section = section
                break
        
        assert projects_section is not None
        assert len(projects_section.items) == 2  # Old items cleared, new ones added
        
        # Verify all items have GENERATED status
        for item in projects_section.items:
            assert item.status == ItemStatus.GENERATED

    async def test_execute_without_existing_section(self, updater_agent, generated_projects_data):
        """Test execute method when projects section doesn't exist."""
        cv_without_projects = StructuredCV(
            sections=[Section(name="Education", items=[])]
        )
        
        agent_input = ProjectsUpdaterAgentInput(
            structured_cv=cv_without_projects,
            generated_projects=generated_projects_data
        )
        
        result = await updater_agent._execute(**agent_input.model_dump())
        
        assert isinstance(result, dict)
        assert "error_messages" in result
        assert "Projects section not found" in str(result["error_messages"])

    async def test_execute_with_empty_generated_data(self, updater_agent, mock_structured_cv):
        """Test execute method with empty generated data."""
        agent_input = ProjectsUpdaterAgentInput(
            structured_cv=mock_structured_cv,
            generated_projects=""
        )
        
        result = await updater_agent._execute(**agent_input.model_dump())
        
        assert isinstance(result, dict)
        assert "error_messages" in result

    async def test_logging_during_execution(self, updater_agent, agent_input, caplog):
        """Test that appropriate logging occurs during execution."""
        import logging
        with caplog.at_level(logging.INFO):
            await updater_agent._execute(**agent_input.model_dump())
        
        assert "Updated Projects section" in caplog.text

    def test_alternative_section_titles(self, updater_agent):
        """Test finding projects section with alternative titles."""
        cv_with_alt_title = StructuredCV(
            sections=[
                Section(title="Key Projects", items=[]),
                Section(title="Education", items=[])
            ]
        )
        
        section = updater_agent._find_projects_section(cv_with_alt_title)
        assert section is not None
        assert section.title == "Key Projects"

    def test_case_insensitive_section_matching(self, updater_agent):
        """Test that section matching is case insensitive."""
        cv_with_lowercase = StructuredCV(
            sections=[
                Section(title="projects", items=[]),
                Section(title="Education", items=[])
            ]
        )
        
        section = updater_agent._find_projects_section(cv_with_lowercase)
        assert section is not None
        assert section.title == "projects"