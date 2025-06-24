"""Unit tests for ParserAgent parsing methods."""

import pytest
from unittest.mock import Mock, patch
from src.agents.parser_agent import ParserAgent
from src.models.data_models import ContentType


class TestParserAgent:
    """Test cases for ParserAgent parsing methods."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_llm_service = Mock()
        mock_vector_store_service = Mock()
        mock_error_recovery_service = Mock()
        mock_progress_tracker = Mock()
        mock_settings = Mock()
        mock_template_manager = Mock()
        self.parser_agent = ParserAgent(
            llm_service=mock_llm_service,
            vector_store_service=mock_vector_store_service,
            error_recovery_service=mock_error_recovery_service,
            progress_tracker=mock_progress_tracker,
            settings=mock_settings,
            template_manager=mock_template_manager,
            name="TestParserAgent",
            description="Test parser agent for unit tests",
        )

    def test_parse_big_10_skills_valid_response(self):
        """Test parsing valid LLM response for Big 10 skills."""
        llm_response = """
        • Python Programming
        • Data Analysis
        • Machine Learning
        • SQL Database Management
        • Project Management
        • Team Leadership
        • Problem Solving
        • Communication Skills
        • Technical Documentation
        • Quality Assurance
        """

        result = self.parser_agent._parse_big_10_skills(llm_response)

        assert len(result) == 10
        assert "Python Programming" in result
        assert "Data Analysis" in result
        assert "Machine Learning" in result

    def test_parse_big_10_skills_with_template_content(self):
        """Test parsing LLM response that contains template instructions."""
        llm_response = """
        [System Instruction] Generate skills based on job description
        You are an expert career advisor
        • Python Programming
        • Data Analysis
        • Machine Learning
        [Additional Context] This is for software engineer role
        • SQL Database Management
        • Project Management
        """

        result = self.parser_agent._parse_big_10_skills(llm_response)

        # Should filter out template content and return valid skills
        assert len(result) == 10  # Should pad with generic skills
        assert "Python Programming" in result
        assert "Data Analysis" in result
        assert "[System Instruction]" not in str(result)

    def test_parse_big_10_skills_insufficient_skills(self):
        """Test parsing when LLM response has fewer than 10 skills."""
        llm_response = """
        • Python Programming
        • Data Analysis
        • Machine Learning
        """

        result = self.parser_agent._parse_big_10_skills(llm_response)

        # Should pad with generic skills to reach 10
        assert len(result) == 10
        assert "Python Programming" in result
        assert "Problem Solving" in result  # Generic skill

    def test_parse_big_10_skills_too_many_skills(self):
        """Test parsing when LLM response has more than 10 skills."""
        llm_response = """
        • Skill 1
        • Skill 2
        • Skill 3
        • Skill 4
        • Skill 5
        • Skill 6
        • Skill 7
        • Skill 8
        • Skill 9
        • Skill 10
        • Skill 11
        • Skill 12
        """

        result = self.parser_agent._parse_big_10_skills(llm_response)

        # Should truncate to exactly 10
        assert len(result) == 10
        assert "Skill 1" in result
        assert "Skill 10" in result
        assert "Skill 11" not in result

    def test_parse_big_10_skills_error_handling(self):
        """Test error handling in _parse_big_10_skills."""
        # Test with None input
        result = self.parser_agent._parse_big_10_skills(None)

        # Should return fallback skills
        assert len(result) == 10
        assert "Problem Solving" in result

    def test_parse_bullet_points_valid_content(self):
        """Test parsing valid content into bullet points."""
        content = """
        • First bullet point
        • Second bullet point
        - Third bullet point
        * Fourth bullet point
        """

        result = self.parser_agent._parse_bullet_points(content)

        assert len(result) == 4
        assert "First bullet point" in result
        assert "Second bullet point" in result
        assert "Third bullet point" in result
        assert "Fourth bullet point" in result

    def test_parse_bullet_points_without_markers(self):
        """Test parsing content without bullet point markers."""
        content = """
        This is a longer line that should be included
        Another meaningful line of content
        Short
        """

        result = self.parser_agent._parse_bullet_points(content)

        # Should include lines longer than 10 characters
        assert "This is a longer line that should be included" in result
        assert "Another meaningful line of content" in result
        assert "Short" not in result  # Too short

    def test_parse_bullet_points_with_excluded_markers(self):
        """Test parsing content with excluded markers."""
        content = """
        • Valid bullet point
        Role: Software Engineer
        Company: Tech Corp
        Dates: 2020-2023
        • Another valid bullet point
        """

        result = self.parser_agent._parse_bullet_points(content)

        assert "Valid bullet point" in result
        assert "Another valid bullet point" in result
        assert "Role: Software Engineer" not in result
        assert "Company: Tech Corp" not in result

    def test_parse_bullet_points_max_limit(self):
        """Test that bullet points are limited to maximum of 5."""
        content = """
        • Point 1
        • Point 2
        • Point 3
        • Point 4
        • Point 5
        • Point 6
        • Point 7
        """

        result = self.parser_agent._parse_bullet_points(content)

        # Should limit to 5 bullet points
        assert len(result) <= 5

    def test_parse_bullet_points_empty_content(self):
        """Test parsing empty or whitespace-only content."""
        result = self.parser_agent._parse_bullet_points("   \n\n   ")

        # Should return the stripped content as single item
        assert len(result) == 0 or result == [""]

    def test_extract_company_from_cv_mapping(self):
        """Test company extraction with role mapping."""
        role_name = "Trainee Data Analyst"
        generation_context = {"original_cv_text": "Some CV text"}

        result = self.parser_agent._extract_company_from_cv(
            role_name, generation_context
        )
        assert result == "STE Smart-Send"

    def test_extract_company_from_cv_fallback(self):
        """Test company extraction fallback logic."""
        role_name = "Software Engineer"
        generation_context = {
            "original_cv_text": "Software Engineer\n[TechCorp] | 2020-2023\n* Developed applications"
        }

        result = self.parser_agent._extract_company_from_cv(
            role_name, generation_context
        )
        assert result == "TechCorp"

    def test_extract_company_from_cv_no_match(self):
        """Test company extraction when no match found."""
        role_name = "Unknown Role"
        generation_context = {"original_cv_text": "Some CV text without the role"}

        result = self.parser_agent._extract_company_from_cv(
            role_name, generation_context
        )
        assert result == "Previous Company"

    def test_parse_cv_text_to_content_item_integration(self):
        """Test the main parsing method integration."""
        cv_text = """
        Software Engineer
        [TechCorp] | 2020-2023
        * Developed web applications
        * Led team of 5 developers

        Data Analyst
        [DataCorp] | 2018-2020
        * Analyzed customer data
        * Created reports
        """
        generation_context = {"original_cv_text": cv_text}

        result = self.parser_agent._parse_cv_text_to_content_item(
            cv_text, generation_context
        )

        from src.models.data_models import Subsection

        assert isinstance(result, Subsection)
        assert result.name == "Professional Experience"
        assert len(result.items) > 0
