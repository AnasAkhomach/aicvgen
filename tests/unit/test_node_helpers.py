"""Unit tests for node_helpers module."""

import pytest
from unittest.mock import MagicMock, patch
from src.orchestration.node_helpers import (
    map_state_to_key_qualifications_input,
    map_state_to_professional_experience_input,
    map_state_to_projects_input,
    map_state_to_executive_summary_input,
    update_cv_with_key_qualifications_data,
    update_cv_with_professional_experience_data,
    update_cv_with_project_data,
    update_cv_with_executive_summary_data
)


class TestMapperFunctions:
    """Test cases for state-to-input mapper functions."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock GlobalState."""
        from src.models.cv_models import StructuredCV, JobDescriptionData, Section, Item, ItemType, ItemStatus
        from uuid import uuid4
        
        # Create a proper StructuredCV instance
        test_item = Item(
            id=uuid4(),
            content="Test content",
            item_type=ItemType.BULLET_POINT,
            status=ItemStatus.PENDING
        )
        test_section = Section(
            id=uuid4(),
            name="Key Qualifications",
            items=[test_item]
        )
        structured_cv = StructuredCV(sections=[test_section])
        
        # Create a proper JobDescriptionData instance
        job_description_data = JobDescriptionData(
            raw_text="Software Engineer position at Test Company",
            job_title="Software Engineer",
            company_name="Test Company",
            skills=["Python", "Testing"],
            responsibilities=["Develop software", "Write tests"]
        )
        
        return {
            "structured_cv": structured_cv,
            "parsed_jd": job_description_data,
            "current_item_id": "test_id",
            "research_data": {"test": "data"},
            "session_id": "test_session"
        }
    
    def test_map_state_to_key_qualifications_input(self, mock_state):
        """Test mapping state to key qualifications input."""
        result = map_state_to_key_qualifications_input(mock_state)
        
        # Verify the result is a KeyQualificationsWriterAgentInput instance
        assert hasattr(result, 'structured_cv')
        assert hasattr(result, 'job_description_data')
        assert hasattr(result, 'current_item_id')
        assert hasattr(result, 'research_findings')
        assert hasattr(result, 'session_id')
        
        # Verify the values are correctly mapped
        assert result.structured_cv == mock_state["structured_cv"]
        assert result.job_description_data == mock_state["parsed_jd"]
        assert result.current_item_id == mock_state["current_item_id"]
        # research_findings should be converted to ResearchFindings model
        from src.models.agent_output_models import ResearchFindings
        assert isinstance(result.research_findings, ResearchFindings)
        assert result.session_id == mock_state["session_id"]
    
    def test_map_state_to_professional_experience_input(self, mock_state):
        """Test mapping state to professional experience input."""
        result = map_state_to_professional_experience_input(mock_state)
        
        # Verify the result is a ProfessionalExperienceWriterAgentInput instance
        assert hasattr(result, 'structured_cv')
        assert hasattr(result, 'job_description_data')
        assert hasattr(result, 'current_item_id')
        assert hasattr(result, 'research_findings')
        assert hasattr(result, 'session_id')
        
        # Verify the values are correctly mapped
        assert result.structured_cv == mock_state["structured_cv"]
        assert result.job_description_data == mock_state["parsed_jd"]
        assert result.current_item_id == mock_state["current_item_id"]
        # research_findings should be converted to ResearchFindings model
        from src.models.agent_output_models import ResearchFindings
        assert isinstance(result.research_findings, ResearchFindings)
        assert result.session_id == mock_state["session_id"]
    
    def test_map_state_to_projects_input(self, mock_state):
        """Test mapping state to projects input."""
        result = map_state_to_projects_input(mock_state)
        
        # Verify the result is a ProjectsWriterAgentInput instance
        assert hasattr(result, 'structured_cv')
        assert hasattr(result, 'job_description_data')
        assert hasattr(result, 'current_item_id')
        assert hasattr(result, 'research_findings')
        assert hasattr(result, 'session_id')
        
        # Verify the values are correctly mapped
        assert result.structured_cv == mock_state["structured_cv"]
        assert result.job_description_data == mock_state["parsed_jd"]
        assert result.current_item_id == mock_state["current_item_id"]
        # research_findings should be converted to ResearchFindings model
        from src.models.agent_output_models import ResearchFindings
        assert isinstance(result.research_findings, ResearchFindings)
        assert result.session_id == mock_state["session_id"]
    
    def test_map_state_to_executive_summary_input(self, mock_state):
        """Test mapping state to executive summary input."""
        from src.models.cv_models import Section, Item, ItemType, ItemStatus
        from uuid import uuid4
        
        # Add more sections to the structured_cv for executive summary mapping
        key_qual_item = Item(
            id=uuid4(),
            content="Strong leadership skills",
            item_type=ItemType.KEY_QUALIFICATION,
            status=ItemStatus.COMPLETED
        )
        key_qual_section = Section(
            id=uuid4(),
            name="Key Qualifications",
            items=[key_qual_item]
        )
        
        exp_item = Item(
            id=uuid4(),
            content="Senior Software Engineer at Tech Corp",
            item_type=ItemType.EXPERIENCE_ROLE_TITLE,
            status=ItemStatus.COMPLETED
        )
        exp_section = Section(
            id=uuid4(),
            name="Professional Experience",
            items=[exp_item]
        )
        
        proj_item = Item(
            id=uuid4(),
            content="Built scalable web application",
            item_type=ItemType.PROJECT_DESCRIPTION_BULLET,
            status=ItemStatus.COMPLETED
        )
        proj_section = Section(
            id=uuid4(),
            name="Project Experience",
            items=[proj_item]
        )
        
        # Update the structured_cv with all sections
        mock_state["structured_cv"].sections = [key_qual_section, exp_section, proj_section]
        
        result = map_state_to_executive_summary_input(mock_state)
        
        # Verify the result is an ExecutiveSummaryWriterAgentInput instance
        assert hasattr(result, 'job_description')
        assert hasattr(result, 'key_qualifications')
        assert hasattr(result, 'professional_experience')
        assert hasattr(result, 'projects')
        assert hasattr(result, 'research_findings')
        
        # Verify the research_findings are correctly mapped (should be dict for ExecutiveSummary)
        assert result.research_findings == mock_state["research_data"]


class TestUpdaterFunctions:
    """Test cases for CV updater functions."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock GlobalState."""
        from src.models.cv_models import StructuredCV, Section, Item, ItemType, ItemStatus
        from uuid import uuid4
        
        # Create a proper StructuredCV instance
        test_item = Item(
            id=uuid4(),
            content="Test content",
            item_type=ItemType.BULLET_POINT,
            status=ItemStatus.PENDING
        )
        test_section = Section(
            id=uuid4(),
            name="Key Qualifications",
            items=[test_item]
        )
        structured_cv = StructuredCV(sections=[test_section])
        
        return {
            "structured_cv": structured_cv,
            "current_item_id": "test_id",
            "other_field": "value"
        }
    
    @patch('src.orchestration.node_helpers.update_item_by_id')
    def test_update_cv_with_key_qualifications_data(self, mock_update, mock_state):
        """Test updating CV with key qualifications data."""
        from src.models.cv_models import ItemStatus
        
        agent_output = {
            "generated_key_qualifications": MagicMock(bullet_points=["Point 1", "Point 2"])
        }
        mock_update.return_value = "updated_cv"
        
        result = update_cv_with_key_qualifications_data(mock_state, agent_output)
        
        mock_update.assert_called_once_with(
            mock_state["structured_cv"],
            "test_id",
            {"content": "• Point 1\n• Point 2", "status": ItemStatus.COMPLETED}
        )
        assert result["structured_cv"] == "updated_cv"
        assert result["last_executed_node"] == "KEY_QUALIFICATIONS_WRITER"
    
    @patch('src.orchestration.node_helpers.update_item_by_id')
    def test_update_cv_with_professional_experience_data(self, mock_update, mock_state):
        """Test updating CV with professional experience data."""
        from src.models.cv_models import ItemStatus
        
        agent_output = {
            "generated_professional_experience": MagicMock(description="Test description", bullet_points=None)
        }
        mock_update.return_value = "updated_cv"
        
        result = update_cv_with_professional_experience_data(mock_state, agent_output)
        
        mock_update.assert_called_once_with(
            mock_state["structured_cv"],
            "test_id",
            {"content": "Test description", "status": ItemStatus.COMPLETED}
        )
        assert result["structured_cv"] == "updated_cv"
        assert result["last_executed_node"] == "PROFESSIONAL_EXPERIENCE_WRITER"
    
    @patch('src.orchestration.node_helpers.update_item_by_id')
    def test_update_cv_with_project_data(self, mock_update, mock_state):
        """Test updating CV with project data."""
        from src.models.cv_models import ItemStatus
        
        agent_output = {
            "generated_projects": MagicMock(bullet_points=["Feature 1", "Feature 2"])
        }
        mock_update.return_value = "updated_cv"
        
        result = update_cv_with_project_data(mock_state, agent_output)
        
        mock_update.assert_called_once_with(
            mock_state["structured_cv"],
            "test_id",
            {"content": "• Feature 1\n• Feature 2", "status": ItemStatus.COMPLETED}
        )
        assert result["structured_cv"] == "updated_cv"
        assert result["last_executed_node"] == "PROJECTS_WRITER"
    
    @patch('src.orchestration.node_helpers.update_item_by_id')
    def test_update_cv_with_executive_summary_data(self, mock_update, mock_state):
        """Test updating CV with executive summary data."""
        from src.models.cv_models import ItemStatus
        
        agent_output = {
            "generated_executive_summary": MagicMock(summary="Test summary")
        }
        mock_update.return_value = "updated_cv"
        
        result = update_cv_with_executive_summary_data(mock_state, agent_output)
        
        mock_update.assert_called_once_with(
            mock_state["structured_cv"],
            "test_id",
            {"content": "Test summary", "status": ItemStatus.COMPLETED}
        )
        assert result["structured_cv"] == "updated_cv"
        assert result["last_executed_node"] == "EXECUTIVE_SUMMARY_WRITER"
    
    def test_update_cv_with_key_qualifications_data_no_output(self, mock_state):
        """Test updating CV with key qualifications when no output is generated."""
        agent_output = {}
        
        result = update_cv_with_key_qualifications_data(mock_state, agent_output)
        
        # Should return state with only last_executed_node updated
        assert result["structured_cv"] == mock_state["structured_cv"]
        assert result["last_executed_node"] == "KEY_QUALIFICATIONS_WRITER"
    
    @patch('src.orchestration.node_helpers.update_item_by_id')
    def test_update_cv_with_project_data_description_fallback(self, mock_update, mock_state):
        """Test updating CV with project data using description fallback."""
        from src.models.cv_models import ItemStatus
        
        agent_output = {
            "generated_projects": MagicMock(bullet_points=None, description="Test description")
        }
        mock_update.return_value = "updated_cv"
        
        result = update_cv_with_project_data(mock_state, agent_output)
        
        mock_update.assert_called_once_with(
            mock_state["structured_cv"],
            "test_id",
            {"content": "Test description", "status": ItemStatus.COMPLETED}
        )
        assert result["structured_cv"] == "updated_cv"
        assert result["last_executed_node"] == "PROJECTS_WRITER"
    
    @patch('src.orchestration.node_helpers.update_item_by_id')
    def test_update_cv_with_project_data_string_fallback(self, mock_update, mock_state):
        """Test updating CV with project data using string fallback."""
        from src.models.cv_models import ItemStatus
        
        agent_output = {
            "generated_projects": "String content"
        }
        mock_update.return_value = "updated_cv"
        
        result = update_cv_with_project_data(mock_state, agent_output)
        
        mock_update.assert_called_once_with(
            mock_state["structured_cv"],
            "test_id",
            {"content": "String content", "status": ItemStatus.COMPLETED}
        )
        assert result["structured_cv"] == "updated_cv"
        assert result["last_executed_node"] == "PROJECTS_WRITER"