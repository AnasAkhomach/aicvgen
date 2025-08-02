"""Unit tests for KeyQualificationsUpdaterAgent."""

import pytest
from unittest.mock import Mock

from src.agents.key_qualifications_updater_agent import KeyQualificationsUpdaterAgent
from src.models.cv_models import StructuredCV, Section, Item, ItemStatus, ItemType
from src.error_handling.exceptions import AgentExecutionError


class TestKeyQualificationsUpdaterAgent:
    """Test cases for KeyQualificationsUpdaterAgent."""

    @pytest.fixture
    def agent(self):
        """Create a KeyQualificationsUpdaterAgent instance for testing."""
        return KeyQualificationsUpdaterAgent(session_id="test_session")

    @pytest.fixture
    def structured_cv(self):
        """Create a test StructuredCV with Key Qualifications section."""
        return StructuredCV(
            sections=[
                Section(
                    name="Executive Summary",
                    items=[
                        Item(
                            content="Experienced software engineer.",
                            status=ItemStatus.GENERATED,
                            item_type=ItemType.EXECUTIVE_SUMMARY_PARA,
                        )
                    ],
                ),
                Section(
                    name="Key Qualifications",
                    items=[],  # Empty initially
                ),
                Section(
                    name="Professional Experience",
                    items=[
                        Item(
                            content="Senior Software Engineer at TechCorp",
                            status=ItemStatus.GENERATED,
                            item_type=ItemType.EXPERIENCE_ROLE_TITLE,
                        )
                    ],
                ),
            ],
            metadata={"version": "1.0"},
            big_10_skills=[],
        )

    @pytest.fixture
    def structured_cv_dict(self, structured_cv):
        """Create a dict representation of StructuredCV for testing."""
        return structured_cv.model_dump()

    @pytest.fixture
    def generated_qualifications(self):
        """Create test generated qualifications."""
        return [
            "Python Development",
            "React Development",
            "Full-Stack Expertise",
            "Cloud Technologies",
            "JavaScript Proficiency",
        ]

    @pytest.mark.asyncio
    async def test_successful_update_with_structured_cv_object(
        self, agent, structured_cv, generated_qualifications
    ):
        """Test successful update with StructuredCV object."""
        result = await agent.run(
            structured_cv=structured_cv,
            generated_key_qualifications=generated_qualifications,
            session_id="test_session",
        )

        assert "structured_cv" in result
        assert "error_messages" not in result

        updated_cv = result["structured_cv"]
        qual_section = next(
            s for s in updated_cv.sections if s.name == "Key Qualifications"
        )

        assert len(qual_section.items) == 5
        assert qual_section.items[0].content == "Python Development"
        assert qual_section.items[0].status == ItemStatus.GENERATED
        assert qual_section.items[0].item_type == ItemType.KEY_QUALIFICATION

    @pytest.mark.asyncio
    async def test_successful_update_with_structured_cv_dict(
        self, agent, structured_cv_dict, generated_qualifications
    ):
        """Test successful update with StructuredCV as dict (from extract_agent_inputs)."""
        result = await agent.run(
            structured_cv=structured_cv_dict,
            generated_key_qualifications=generated_qualifications,
            session_id="test_session",
        )

        assert "structured_cv" in result
        assert "error_messages" not in result

        updated_cv = result["structured_cv"]
        qual_section = next(
            s for s in updated_cv.sections if s.name == "Key Qualifications"
        )

        assert len(qual_section.items) == 5
        assert qual_section.items[0].content == "Python Development"
        assert qual_section.items[0].status == ItemStatus.GENERATED
        assert qual_section.items[0].item_type == ItemType.KEY_QUALIFICATION

    @pytest.mark.asyncio
    async def test_missing_structured_cv(self, agent, generated_qualifications):
        """Test error when structured_cv is missing."""
        result = await agent.run(
            generated_key_qualifications=generated_qualifications,
            session_id="test_session",
        )

        assert "error_messages" in result
        assert "Missing required input: structured_cv" in result["error_messages"][0]

    @pytest.mark.asyncio
    async def test_missing_generated_qualifications(self, agent, structured_cv):
        """Test error when generated_key_qualifications is missing."""
        result = await agent.run(structured_cv=structured_cv, session_id="test_session")

        assert "error_messages" in result
        assert (
            "Missing required input: generated_key_qualifications"
            in result["error_messages"][0]
        )

    @pytest.mark.asyncio
    async def test_empty_generated_qualifications(self, agent, structured_cv):
        """Test error when generated_key_qualifications is empty."""
        result = await agent.run(
            structured_cv=structured_cv,
            generated_key_qualifications=[],
            session_id="test_session",
        )

        assert "error_messages" in result
        assert (
            "generated_key_qualifications cannot be empty"
            in result["error_messages"][0]
        )

    @pytest.mark.asyncio
    async def test_invalid_structured_cv_type(self, agent, generated_qualifications):
        """Test error when structured_cv is invalid type (decorator handles validation)."""
        result = await agent.run(
            structured_cv="invalid_type",
            generated_key_qualifications=generated_qualifications,
            session_id="test_session",
        )

        assert "error_messages" in result
        # The decorator logs a warning for non-dict fields, then the agent fails when trying to access .sections
        assert "'str' object has no attribute 'sections'" in result["error_messages"][0]

    @pytest.mark.asyncio
    async def test_invalid_generated_qualifications_type(self, agent, structured_cv):
        """Test error when generated_key_qualifications is not a list."""
        result = await agent.run(
            structured_cv=structured_cv,
            generated_key_qualifications="not_a_list",
            session_id="test_session",
        )

        assert "error_messages" in result
        assert (
            "generated_key_qualifications must be a list" in result["error_messages"][0]
        )

    @pytest.mark.asyncio
    async def test_missing_key_qualifications_section(
        self, agent, generated_qualifications
    ):
        """Test error when Key Qualifications section is missing from CV."""
        cv_without_qual_section = StructuredCV(
            sections=[
                Section(
                    name="Executive Summary",
                    items=[],
                ),
                Section(
                    name="Professional Experience",
                    items=[],
                ),
            ],
            metadata={"version": "1.0"},
            big_10_skills=[],
        )

        result = await agent.run(
            structured_cv=cv_without_qual_section,
            generated_key_qualifications=generated_qualifications,
            session_id="test_session",
        )

        assert "error_messages" in result
        assert "Key Qualifications section not found" in result["error_messages"][0]

    @pytest.mark.asyncio
    async def test_replaces_existing_qualifications(
        self, agent, structured_cv, generated_qualifications
    ):
        """Test that existing qualifications are replaced with new ones."""
        # Add some existing qualifications
        qual_section = next(
            s for s in structured_cv.sections if s.name == "Key Qualifications"
        )
        qual_section.items = [
            Item(
                content="Old Qualification 1",
                status=ItemStatus.GENERATED,
                item_type=ItemType.KEY_QUALIFICATION,
            ),
            Item(
                content="Old Qualification 2",
                status=ItemStatus.GENERATED,
                item_type=ItemType.KEY_QUALIFICATION,
            ),
        ]

        result = await agent.run(
            structured_cv=structured_cv,
            generated_key_qualifications=generated_qualifications,
            session_id="test_session",
        )

        assert "structured_cv" in result
        updated_cv = result["structured_cv"]
        updated_qual_section = next(
            s for s in updated_cv.sections if s.name == "Key Qualifications"
        )

        # Should have new qualifications, not old ones
        assert len(updated_qual_section.items) == 5
        assert updated_qual_section.items[0].content == "Python Development"
        assert "Old Qualification 1" not in [
            item.content for item in updated_qual_section.items
        ]

    def test_validate_inputs_with_valid_data(
        self, agent, structured_cv, generated_qualifications
    ):
        """Test _validate_inputs with valid data."""
        kwargs = {
            "structured_cv": structured_cv,
            "generated_key_qualifications": generated_qualifications,
        }

        # Should not raise any exception
        agent._validate_inputs(kwargs)

    def test_validate_inputs_with_dict_conversion(
        self, agent, structured_cv_dict, generated_qualifications
    ):
        """Test _validate_inputs with dict (decorator handles conversion)."""
        kwargs = {
            "structured_cv": structured_cv_dict,
            "generated_key_qualifications": generated_qualifications,
        }

        # Should not raise any exception - decorator handles conversion
        agent._validate_inputs(kwargs)
        # The dict remains as dict in _validate_inputs since decorator handles conversion
        assert isinstance(kwargs["structured_cv"], dict)
