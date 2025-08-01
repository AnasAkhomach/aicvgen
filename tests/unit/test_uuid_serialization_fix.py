"""Tests for UUID serialization fix in workflow manager.

This test verifies that UUID objects in Pydantic models are properly
serialized to JSON strings without causing serialization errors.
"""

import json
import pytest
from uuid import UUID, uuid4

from src.core.managers.workflow_manager import _serialize_for_json
from src.models.cv_models import StructuredCV, Section, Item, ItemStatus


class TestUUIDSerializationFix:
    """Test UUID serialization in workflow manager."""

    def test_serialize_uuid_object_directly(self):
        """Test that UUID objects are converted to strings."""
        test_uuid = uuid4()
        result = _serialize_for_json(test_uuid)

        assert isinstance(result, str)
        assert result == str(test_uuid)
        # Verify it's a valid UUID string
        UUID(result)  # Should not raise an exception

    def test_serialize_structured_cv_with_uuids(self):
        """Test that StructuredCV with UUID fields serializes properly."""
        # Create a StructuredCV with sections and items that have UUID fields
        cv = StructuredCV.create_empty()

        # Add a section with UUID
        section = Section(name="Experience", content_type="DYNAMIC")

        # Add an item with UUID
        item = Item(content="Software Engineer at TechCorp", status=ItemStatus.INITIAL)
        section.items.append(item)
        cv.sections.append(section)

        # Serialize the CV
        result = _serialize_for_json(cv)

        # Verify it can be JSON serialized without errors
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Verify the structure is preserved
        assert "__pydantic_model__" in result
        assert "data" in result

        # Verify UUID fields are strings in the serialized data
        cv_data = result["data"]
        assert isinstance(cv_data["id"], str)

        # Check section UUIDs (create_empty creates 5 standard sections)
        sections_data = cv_data["sections"]
        assert len(sections_data) == 6  # 5 standard + 1 added
        # Check that all section IDs are strings
        for section_data in sections_data:
            assert isinstance(section_data["id"], str)

        # Find the Experience section and check its item UUIDs
        experience_section = None
        for section_data in sections_data:
            if section_data["name"] == "Experience":
                experience_section = section_data
                break

        assert experience_section is not None
        items_data = experience_section["items"]
        assert len(items_data) == 1
        assert isinstance(items_data[0]["id"], str)

    def test_serialize_nested_structure_with_uuids(self):
        """Test serialization of nested structures containing UUIDs."""
        test_data = {
            "session_id": uuid4(),
            "nested": {"item_id": uuid4(), "items": [uuid4(), uuid4()]},
            "regular_string": "test",
            "number": 42,
        }

        result = _serialize_for_json(test_data)

        # Verify all UUIDs are converted to strings
        assert isinstance(result["session_id"], str)
        assert isinstance(result["nested"]["item_id"], str)
        assert all(isinstance(item, str) for item in result["nested"]["items"])

        # Verify other data types are preserved
        assert result["regular_string"] == "test"
        assert result["number"] == 42

        # Verify it can be JSON serialized
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_json_serialization_no_longer_fails(self):
        """Test that the original error 'Object of type UUID is not JSON serializable' is fixed."""
        # Create a structure that would have caused the original error
        cv = StructuredCV.create_empty()
        section = Section(name="Test Section")
        cv.sections.append(section)

        # This should not raise 'Object of type UUID is not JSON serializable'
        serialized = _serialize_for_json(cv)

        # This should not raise any JSON serialization errors
        json_string = json.dumps(serialized)

        # Verify we can parse it back
        parsed = json.loads(json_string)
        assert parsed is not None
        assert "__pydantic_model__" in parsed
