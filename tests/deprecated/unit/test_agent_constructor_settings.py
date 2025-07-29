"""Test to verify that the container settings injection fix works correctly."""

from typing import Any, Union

import pytest


class TestAgentSettingsFix:
    """Test that all agents properly handle settings parameter."""

    # Note: We can't test AgentBase directly as it's an abstract class with abstract methods.
    # Instead, we test concrete agent implementations to ensure they handle settings correctly.

    def test_cleaning_agent_constructor_compatibility(self):
        """Test that CleaningAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.cleaning_agent import CleaningAgent

        mock_llm_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = CleaningAgent(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "CleaningAgent"
        assert agent.session_id == "test_session"

    def test_key_qualifications_writer_agent_constructor_compatibility(self):
        """Test that KeyQualificationsWriterAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.key_qualifications_writer_agent import (
            KeyQualificationsWriterAgent,
        )

        mock_llm_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = KeyQualificationsWriterAgent(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "KeyQualificationsWriter"
        assert agent.session_id == "test_session"

    def test_professional_experience_writer_agent_constructor_compatibility(self):
        """Test that ProfessionalExperienceWriterAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.professional_experience_writer_agent import (
            ProfessionalExperienceWriterAgent,
        )

        mock_llm_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = ProfessionalExperienceWriterAgent(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "ProfessionalExperienceWriter"
        assert agent.session_id == "test_session"

    def test_projects_writer_agent_constructor_compatibility(self):
        """Test that ProjectsWriterAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.projects_writer_agent import ProjectsWriterAgent

        mock_llm_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = ProjectsWriterAgent(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "ProjectsWriter"
        assert agent.session_id == "test_session"

    def test_executive_summary_writer_agent_constructor_compatibility(self):
        """Test that ExecutiveSummaryWriterAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.executive_summary_writer_agent import (
            ExecutiveSummaryWriterAgent,
        )

        mock_llm_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = ExecutiveSummaryWriterAgent(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "ExecutiveSummaryWriter"
        assert agent.session_id == "test_session"

    def test_quality_assurance_agent_constructor_compatibility(self):
        """Test that QualityAssuranceAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.quality_assurance_agent import QualityAssuranceAgent

        mock_llm_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = QualityAssuranceAgent(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "QualityAssuranceAgent"
        assert agent.session_id == "test_session"

    def test_formatter_agent_constructor_compatibility(self):
        """Test that FormatterAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.formatter_agent import FormatterAgent

        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = FormatterAgent(
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "FormatterAgent"
        assert agent.session_id == "test_session"

    def test_research_agent_constructor_compatibility(self):
        """Test that ResearchAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.research_agent import ResearchAgent

        mock_llm_service = Mock()
        mock_vector_store_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = ResearchAgent(
            llm_service=mock_llm_service,
            vector_store_service=mock_vector_store_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "ResearchAgent"
        assert agent.session_id == "test_session"

    def test_enhanced_content_writer_agent_constructor_compatibility(self):
        """Test that EnhancedContentWriterAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.enhanced_content_writer import EnhancedContentWriterAgent

        mock_llm_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = EnhancedContentWriterAgent(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "EnhancedContentWriter"
        assert agent.session_id == "test_session"

    def test_job_description_parser_agent_constructor_compatibility(self):
        """Test that JobDescriptionParserAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.job_description_parser_agent import JobDescriptionParserAgent

        mock_llm_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = JobDescriptionParserAgent(
            llm_service=mock_llm_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "JobDescriptionParserAgent"
        assert agent.session_id == "test_session"

    def test_user_cv_parser_agent_constructor_compatibility(self):
        """Test that UserCVParserAgent constructor is compatible with settings."""
        from unittest.mock import Mock

        from src.agents.user_cv_parser_agent import UserCVParserAgent

        mock_llm_service = Mock()
        mock_vector_store_service = Mock()
        mock_template_manager = Mock()
        settings = {"test_setting": "test_value"}

        agent = UserCVParserAgent(
            llm_service=mock_llm_service,
            vector_store_service=mock_vector_store_service,
            template_manager=mock_template_manager,
            settings=settings,
            session_id="test_session",
        )

        assert agent.settings == settings
        assert agent.name == "UserCVParserAgent"
        assert agent.session_id == "test_session"
