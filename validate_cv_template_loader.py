#!/usr/bin/env python3
"""Simple validation script for CVTemplateLoaderService without pytest dependency."""

import sys
import tempfile
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

try:
    from services.cv_template_loader_service import CVTemplateLoaderService
    from models.cv_models import StructuredCV
    print("✓ Successfully imported CVTemplateLoaderService and StructuredCV")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of the service."""
    print("\n=== Testing Basic Functionality ===")
    
    # Create a test markdown template
    markdown_content = """
# CV Template

## Personal Information
Some personal info content here.

### Contact Details
Email and phone details.

### Address
Home address information.

## Experience
Work experience section.

### Software Engineer
Details about software engineering role.

### Project Manager
Details about project management role.

## Education
Education background.

### University Degree
Bachelor's degree information.
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(markdown_content)
        temp_path = f.name
    
    try:
        # Load the template
        result = CVTemplateLoaderService.load_from_markdown(temp_path)
        
        # Basic validation
        assert isinstance(result, StructuredCV), f"Expected StructuredCV, got {type(result)}"
        print("✓ Successfully created StructuredCV object")
        
        # Check sections
        assert len(result.sections) == 3, f"Expected 3 sections, got {len(result.sections)}"
        print("✓ Correct number of sections parsed")
        
        section_names = [section.name for section in result.sections]
        expected_sections = ["Personal Information", "Experience", "Education"]
        for expected in expected_sections:
            assert expected in section_names, f"Missing section: {expected}"
        print("✓ All expected sections found")
        
        # Check subsections
        personal_section = next(s for s in result.sections if s.name == "Personal Information")
        assert len(personal_section.subsections) == 2, f"Expected 2 subsections in Personal Information, got {len(personal_section.subsections)}"
        print("✓ Correct number of subsections in Personal Information")
        
        # Check that items are empty
        for section in result.sections:
            assert section.items == [], f"Section {section.name} should have empty items list"
            for subsection in section.subsections:
                assert subsection.items == [], f"Subsection {subsection.name} should have empty items list"
        print("✓ All items lists are properly initialized as empty")
        
        # Check metadata
        assert result.metadata.created_by == 'cv_template_loader', "Incorrect metadata.created_by"
        assert str(temp_path) in result.metadata.source_file, "Source file not in metadata"
        print("✓ Metadata properly set")
        
        print("✓ Basic functionality test PASSED")
        
    except Exception as e:
        print(f"✗ Basic functionality test FAILED: {e}")
        traceback.print_exc()
        return False
    finally:
        Path(temp_path).unlink()
    
    return True

def test_error_handling():
    """Test error handling."""
    print("\n=== Testing Error Handling ===")
    
    # Test file not found
    try:
        CVTemplateLoaderService.load_from_markdown("/nonexistent/file.md")
        print("✗ Should have raised FileNotFoundError")
        return False
    except FileNotFoundError:
        print("✓ Correctly handles missing files")
    except Exception as e:
        print(f"✗ Unexpected exception for missing file: {e}")
        return False
    
    # Test empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write("   \n\t  \n  ")  # Only whitespace
        temp_path = f.name
    
    try:
        CVTemplateLoaderService.load_from_markdown(temp_path)
        print("✗ Should have raised ValueError for empty file")
        return False
    except ValueError as e:
        if "empty" in str(e).lower():
            print("✓ Correctly handles empty files")
        else:
            print(f"✗ Wrong error message for empty file: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected exception for empty file: {e}")
        return False
    finally:
        Path(temp_path).unlink()
    
    # Test no sections
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write("# Just a title\n\nSome content without sections.")
        temp_path = f.name
    
    try:
        CVTemplateLoaderService.load_from_markdown(temp_path)
        print("✗ Should have raised ValueError for no sections")
        return False
    except ValueError as e:
        if "no valid sections" in str(e).lower():
            print("✓ Correctly handles files with no sections")
        else:
            print(f"✗ Wrong error message for no sections: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected exception for no sections: {e}")
        return False
    finally:
        Path(temp_path).unlink()
    
    print("✓ Error handling test PASSED")
    return True

def test_regex_patterns():
    """Test regex patterns."""
    print("\n=== Testing Regex Patterns ===")
    
    section_pattern = CVTemplateLoaderService.SECTION_PATTERN
    subsection_pattern = CVTemplateLoaderService.SUBSECTION_PATTERN
    
    # Test section pattern
    valid_sections = ["## Valid Section", "##Another Section", "##  Spaced Section  "]
    invalid_sections = ["# Single hash", "### Triple hash", "Text ## in middle"]
    
    for valid in valid_sections:
        assert section_pattern.search(valid), f"Should match: {valid}"
    
    for invalid in invalid_sections:
        assert not section_pattern.search(invalid), f"Should not match: {invalid}"
    
    print("✓ Section pattern works correctly")
    
    # Test subsection pattern
    valid_subsections = ["### Valid Subsection", "###Another Subsection", "###  Spaced Subsection  "]
    invalid_subsections = ["## Double hash", "#### Quad hash", "Text ### in middle"]
    
    for valid in valid_subsections:
        assert subsection_pattern.search(valid), f"Should match: {valid}"
    
    for invalid in invalid_subsections:
        assert not subsection_pattern.search(invalid), f"Should not match: {invalid}"
    
    print("✓ Subsection pattern works correctly")
    print("✓ Regex patterns test PASSED")
    return True

def main():
    """Run all validation tests."""
    print("CVTemplateLoaderService Validation")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_error_handling,
        test_regex_patterns
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests PASSED! CVTemplateLoaderService is working correctly.")
        return 0
    else:
        print("❌ Some tests FAILED. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())