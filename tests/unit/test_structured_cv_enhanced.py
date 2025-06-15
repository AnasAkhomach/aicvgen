"""Unit tests for enhanced StructuredCV class methods.

Tests the new methods added to StructuredCV class for Task 1.1:
- get_section_by_name
- find_section_by_id
- update_item_content
- update_item_status
- get_items_by_status
- to_content_data
- update_from_content
"""

import pytest
from uuid import uuid4
from src.models.data_models import (
    StructuredCV, Section, Subsection, Item, ItemStatus, ItemType
)


class TestStructuredCVEnhanced:
    """Test suite for enhanced StructuredCV methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cv = StructuredCV()
        
        # Create test sections with items
        self.section1 = Section(
            name="Executive Summary",
            items=[
                Item(content="Summary item 1", status=ItemStatus.INITIAL),
                Item(content="Summary item 2", status=ItemStatus.GENERATED)
            ]
        )
        
        self.section2 = Section(
            name="Professional Experience",
            items=[
                Item(content="Experience item 1", status=ItemStatus.USER_MODIFIED)
            ],
            subsections=[
                Subsection(
                    name="Senior Developer",
                    items=[
                        Item(content="Subsection item 1", status=ItemStatus.INITIAL),
                        Item(content="Subsection item 2", status=ItemStatus.GENERATED)
                    ]
                )
            ]
        )
        
        self.cv.sections = [self.section1, self.section2]
    
    def test_get_section_by_name_found(self):
        """Test getting a section by name when it exists."""
        section = self.cv.get_section_by_name("Executive Summary")
        assert section is not None
        assert section.name == "Executive Summary"
        assert len(section.items) == 2
    
    def test_get_section_by_name_not_found(self):
        """Test getting a section by name when it doesn't exist."""
        section = self.cv.get_section_by_name("Nonexistent Section")
        assert section is None
    
    def test_find_section_by_id_found(self):
        """Test finding a section by ID when it exists."""
        section_id = str(self.section1.id)
        section = self.cv.find_section_by_id(section_id)
        assert section is not None
        assert str(section.id) == section_id
        assert section.name == "Executive Summary"
    
    def test_find_section_by_id_not_found(self):
        """Test finding a section by ID when it doesn't exist."""
        fake_id = str(uuid4())
        section = self.cv.find_section_by_id(fake_id)
        assert section is None
    
    def test_update_item_content_success(self):
        """Test updating item content successfully."""
        item_id = str(self.section1.items[0].id)
        new_content = "Updated summary content"
        
        result = self.cv.update_item_content(item_id, new_content)
        assert result is True
        
        # Verify the content was updated
        item, _, _ = self.cv.find_item_by_id(item_id)
        assert item.content == new_content
    
    def test_update_item_content_not_found(self):
        """Test updating item content when item doesn't exist."""
        fake_id = str(uuid4())
        result = self.cv.update_item_content(fake_id, "New content")
        assert result is False
    
    def test_update_item_status_success(self):
        """Test updating item status successfully."""
        item_id = str(self.section1.items[0].id)
        new_status = ItemStatus.USER_ACCEPTED
        
        result = self.cv.update_item_status(item_id, new_status)
        assert result is True
        
        # Verify the status was updated
        item, _, _ = self.cv.find_item_by_id(item_id)
        assert item.status == new_status
    
    def test_update_item_status_not_found(self):
        """Test updating item status when item doesn't exist."""
        fake_id = str(uuid4())
        result = self.cv.update_item_status(fake_id, ItemStatus.USER_ACCEPTED)
        assert result is False
    
    def test_get_items_by_status_initial(self):
        """Test getting items by INITIAL status."""
        items = self.cv.get_items_by_status(ItemStatus.INITIAL)
        assert len(items) == 2  # One from section1, one from subsection
        
        # Verify all items have INITIAL status
        for item in items:
            assert item.status == ItemStatus.INITIAL
    
    def test_get_items_by_status_generated(self):
        """Test getting items by GENERATED status."""
        items = self.cv.get_items_by_status(ItemStatus.GENERATED)
        assert len(items) == 2  # One from section1, one from subsection
        
        # Verify all items have GENERATED status
        for item in items:
            assert item.status == ItemStatus.GENERATED
    
    def test_get_items_by_status_no_matches(self):
        """Test getting items by status with no matches."""
        items = self.cv.get_items_by_status(ItemStatus.GENERATION_FAILED)
        assert len(items) == 0
    
    def test_to_content_data_conversion(self):
        """Test converting StructuredCV to ContentData format."""
        content_data = self.cv.to_content_data()
        
        # Check structure
        expected_keys = [
            "personal_info", "executive_summary", "professional_experience",
            "key_qualifications", "projects", "education"
        ]
        for key in expected_keys:
            assert key in content_data
        
        # Check executive summary conversion
        assert len(content_data["executive_summary"]) == 2
        assert "Summary item 1" in content_data["executive_summary"]
        assert "Summary item 2" in content_data["executive_summary"]
        
        # Check professional experience conversion
        assert len(content_data["professional_experience"]) == 1
        assert "Experience item 1" in content_data["professional_experience"]
    
    def test_update_from_content_success(self):
        """Test updating StructuredCV from ContentData format."""
        content_data = {
            "personal_info": {"name": "John Doe", "email": "john@example.com"},
            "executive_summary": ["New summary 1", "New summary 2"],
            "professional_experience": ["New experience 1"],
            "key_qualifications": ["Skill 1", "Skill 2"],
            "projects": ["Project 1"],
            "education": ["Degree 1"]
        }
        
        result = self.cv.update_from_content(content_data)
        assert result is True
        
        # Verify personal info was updated
        assert self.cv.metadata["personal_info"]["name"] == "John Doe"
        assert self.cv.metadata["personal_info"]["email"] == "john@example.com"
        
        # Verify sections were recreated
        assert len(self.cv.sections) == 5  # All 5 content sections
        
        # Check executive summary section
        exec_section = self.cv.get_section_by_name("Executive Summary")
        assert exec_section is not None
        assert len(exec_section.items) == 2
        assert exec_section.items[0].content == "New summary 1"
        assert exec_section.items[1].content == "New summary 2"
        
        # Check all items have INITIAL status
        for section in self.cv.sections:
            for item in section.items:
                assert item.status == ItemStatus.INITIAL
    
    def test_update_from_content_empty_sections_skipped(self):
        """Test that empty sections are skipped during update."""
        content_data = {
            "executive_summary": ["Summary 1"],
            "professional_experience": [],  # Empty
            "key_qualifications": [""],  # Empty string
            "projects": ["Project 1"],
            "education": []  # Empty
        }
        
        result = self.cv.update_from_content(content_data)
        assert result is True
        
        # Should only have 2 sections (executive_summary and projects)
        assert len(self.cv.sections) == 2
        
        section_names = [section.name for section in self.cv.sections]
        assert "Executive Summary" in section_names
        assert "Projects" in section_names
        assert "Professional Experience" not in section_names
        assert "Key Qualifications" not in section_names
        assert "Education" not in section_names
    
    def test_update_from_content_invalid_data(self):
        """Test update_from_content with invalid data structure."""
        # This should not raise an exception but return False
        invalid_data = "not a dictionary"
        result = self.cv.update_from_content(invalid_data)
        assert result is False