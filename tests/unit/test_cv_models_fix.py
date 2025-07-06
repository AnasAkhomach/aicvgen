"""Tests for CV models fix to ensure required sections are always present."""

import pytest
from uuid import uuid4
from src.models.cv_models import StructuredCV, Section, ItemStatus, Item


class TestStructuredCVRequiredSections:
    """Test cases for ensuring required sections in StructuredCV."""

    def test_ensure_required_sections_adds_missing_sections(self):
        """Test that ensure_required_sections adds missing required sections."""
        # Create a CV with only one section
        existing_section = Section(
            id=uuid4(),
            name="Professional Experience",
            content_type="DYNAMIC",
            order=0,
            status=ItemStatus.INITIAL,
            subsections=[],
            items=[]
        )
        
        cv = StructuredCV(sections=[existing_section])
        
        # Ensure required sections
        cv.ensure_required_sections()
        
        # Check that all required sections are present
        required_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
            "Education"
        ]
        
        section_names = {section.name for section in cv.sections}
        
        for required_section in required_sections:
            assert required_section in section_names, f"Missing required section: {required_section}"
        
        # Should have 5 sections total
        assert len(cv.sections) == 5

    def test_ensure_required_sections_preserves_existing_sections(self):
        """Test that ensure_required_sections preserves existing sections."""
        # Create a CV with all required sections already present
        sections = []
        required_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
            "Education"
        ]
        
        for i, section_name in enumerate(required_sections):
            test_item = Item(
                id=uuid4(),
                content=f"Test item for {section_name}",
                status=ItemStatus.INITIAL
            )
            section = Section(
                id=uuid4(),
                name=section_name,
                content_type="DYNAMIC",
                order=i,
                status=ItemStatus.INITIAL,
                subsections=[],
                items=[test_item]
            )
            sections.append(section)
        
        cv = StructuredCV(sections=sections)
        original_section_count = len(cv.sections)
        
        # Ensure required sections
        cv.ensure_required_sections()
        
        # Should not add any new sections
        assert len(cv.sections) == original_section_count
        
        # Should preserve existing content
        for section in cv.sections:
            assert len(section.items) == 1
            assert section.items[0].content == f"Test item for {section.name}"

    def test_ensure_required_sections_with_empty_cv(self):
        """Test that ensure_required_sections works with completely empty CV."""
        cv = StructuredCV(sections=[])
        
        # Ensure required sections
        cv.ensure_required_sections()
        
        # Check that all required sections are present
        required_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
            "Education"
        ]
        
        section_names = {section.name for section in cv.sections}
        
        for required_section in required_sections:
            assert required_section in section_names, f"Missing required section: {required_section}"
        
        # Should have 5 sections total
        assert len(cv.sections) == 5
        
        # All sections should be empty initially
        for section in cv.sections:
            assert len(section.items) == 0
            assert len(section.subsections) == 0
            assert section.status == ItemStatus.INITIAL

    def test_create_empty_includes_all_required_sections(self):
        """Test that create_empty method includes all required sections."""
        cv = StructuredCV.create_empty()
        
        required_sections = [
            "Executive Summary",
            "Key Qualifications",
            "Professional Experience",
            "Project Experience",
            "Education"
        ]
        
        section_names = {section.name for section in cv.sections}
        
        for required_section in required_sections:
            assert required_section in section_names, f"Missing required section: {required_section}"
        
        assert len(cv.sections) == 5