"""Unit tests for section.name vs section.title consistency (Task NI-01).

This test ensures that Section objects use the correct 'name' attribute
and prevents regressions where 'title' might be incorrectly accessed.
"""

import pytest
from src.models.data_models import Section, Item, ItemType, ItemStatus
from uuid import uuid4


class TestSectionNameConsistency:
    """Test cases for section name attribute consistency."""

    def test_section_has_name_attribute(self):
        """Test that Section objects have 'name' attribute."""
        section = Section(name="Professional Experience")
        
        # Verify 'name' attribute exists and works
        assert hasattr(section, 'name')
        assert section.name == "Professional Experience"
        
        # Verify 'title' attribute does NOT exist
        assert not hasattr(section, 'title')

    def test_section_name_attribute_access(self):
        """Test that section.name can be accessed without errors."""
        section = Section(
            name="Key Qualifications",
            items=[
                Item(
                    content="Python programming",
                    item_type=ItemType.KEY_QUALIFICATION,
                    status=ItemStatus.INITIAL
                )
            ]
        )
        
        # This should work without any AttributeError
        section_name = section.name
        assert section_name == "Key Qualifications"

    def test_section_title_attribute_does_not_exist(self):
        """Test that accessing section.title raises AttributeError."""
        section = Section(name="Education")
        
        # This should raise AttributeError to prevent incorrect usage
        with pytest.raises(AttributeError, match="'Section' object has no attribute 'title'"):
            _ = section.title

    def test_section_name_in_template_context(self):
        """Test that section.name works in template-like contexts."""
        section = Section(name="Professional Experience")
        
        # Simulate template usage like in PDF generation
        template_context = f"This is for the {section.name} section of a CV"
        expected = "This is for the Professional Experience section of a CV"
        
        assert template_context == expected

    def test_multiple_sections_name_access(self):
        """Test that multiple sections can have their names accessed correctly."""
        sections = [
            Section(name="Executive Summary"),
            Section(name="Professional Experience"),
            Section(name="Key Qualifications"),
            Section(name="Education")
        ]
        
        expected_names = [
            "Executive Summary",
            "Professional Experience", 
            "Key Qualifications",
            "Education"
        ]
        
        actual_names = [section.name for section in sections]
        assert actual_names == expected_names

    def test_section_name_modification(self):
        """Test that section.name can be modified correctly."""
        section = Section(name="Original Name")
        
        # Modify the name
        section.name = "Updated Name"
        
        # Verify the change
        assert section.name == "Updated Name"