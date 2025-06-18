"""Unit tests for LaTeX character escaping utilities."""

import pytest
from src.utils.latex_utils import escape_latex, recursively_escape_latex


class TestEscapeLatex:
    """Test cases for the escape_latex function."""
    
    def test_escape_basic_characters(self):
        """Test escaping of basic LaTeX special characters."""
        input_text = "Hello _world_ & Co. 100% #1 {awesome}"
        expected = "Hello \_world\_ \& Co. 100\% \#1 \{awesome\}"
        result = escape_latex(input_text)
        assert result == expected
    
    def test_escape_all_special_characters(self):
        """Test escaping of all LaTeX special characters."""
        input_text = "Test & % $ # ^ _ { } ~ \\ chars"
        expected = "Test \& \% \$ \# \textasciicircum{} \_ \{ \} \textasciitilde{} \textbackslash{} chars"
        result = escape_latex(input_text)
        assert result == expected
    
    def test_escape_empty_string(self):
        """Test escaping of empty string."""
        result = escape_latex("")
        assert result == ""
    
    def test_escape_no_special_characters(self):
        """Test escaping of string with no special characters."""
        input_text = "Hello World 123"
        result = escape_latex(input_text)
        assert result == input_text
    
    def test_escape_non_string_input(self):
        """Test that non-string inputs are returned unchanged."""
        assert escape_latex(123) == 123
        assert escape_latex(None) is None
        assert escape_latex([1, 2, 3]) == [1, 2, 3]
        assert escape_latex({"key": "value"}) == {"key": "value"}
    
    def test_escape_backslash_first(self):
        """Test that backslashes are escaped first to avoid double-escaping."""
        input_text = "\\&"
        expected = "\textbackslash{}\&"
        result = escape_latex(input_text)
        assert result == expected
    
    def test_escape_multiple_occurrences(self):
        """Test escaping when special characters appear multiple times."""
        input_text = "100% & 200% & 300%"
        expected = "100\% \& 200\% \& 300\%"
        result = escape_latex(input_text)
        assert result == expected
    
    def test_escape_programming_context(self):
        """Test escaping in programming-related contexts."""
        input_text = "C++ developer with 90% success rate"
        expected = "C\textasciicircum{}\textasciicircum{} developer with 90\% success rate"
        result = escape_latex(input_text)
        assert result == expected
    
    def test_escape_currency_and_math(self):
        """Test escaping of currency and mathematical symbols."""
        input_text = "Saved $1000 & increased efficiency by 25%"
        expected = "Saved \$1000 \& increased efficiency by 25\%"
        result = escape_latex(input_text)
        assert result == expected


class TestRecursivelyEscapeLatex:
    """Test cases for the recursively_escape_latex function."""
    
    def test_escape_simple_dict(self):
        """Test escaping of a simple dictionary."""
        input_data = {
            "name": "John & Jane",
            "company": "Tech Corp 100%"
        }
        expected = {
            "name": "John \& Jane",
            "company": "Tech Corp 100\%"
        }
        result = recursively_escape_latex(input_data)
        assert result == expected
    
    def test_escape_simple_list(self):
        """Test escaping of a simple list."""
        input_data = ["C++", "Data Analysis 100%", "Team Lead & Manager"]
        expected = ["C\textasciicircum{}\textasciicircum{}", "Data Analysis 100\%", "Team Lead \& Manager"]
        result = recursively_escape_latex(input_data)
        assert result == expected
    
    def test_escape_nested_structure(self):
        """Test escaping of nested dictionary and list structure."""
        input_data = {
            "name": "John & Jane",
            "skills": ["C++", "Data Analysis 100%"],
            "metadata": {
                "score": 95,
                "notes": "Top #1 candidate"
            }
        }
        expected = {
            "name": "John \& Jane",
            "skills": ["C\textasciicircum{}\textasciicircum{}", "Data Analysis 100\%"],
            "metadata": {
                "score": 95,
                "notes": "Top \#1 candidate"
            }
        }
        result = recursively_escape_latex(input_data)
        assert result == expected
    
    def test_escape_preserves_non_strings(self):
        """Test that non-string values are preserved unchanged."""
        input_data = {
            "name": "John & Jane",
            "age": 30,
            "active": True,
            "score": 95.5,
            "notes": None,
            "skills": ["Python", 42, True, None]
        }
        expected = {
            "name": "John \& Jane",
            "age": 30,
            "active": True,
            "score": 95.5,
            "notes": None,
            "skills": ["Python", 42, True, None]
        }
        result = recursively_escape_latex(input_data)
        assert result == expected
    
    def test_escape_empty_structures(self):
        """Test escaping of empty data structures."""
        assert recursively_escape_latex({}) == {}
        assert recursively_escape_latex([]) == []
        assert recursively_escape_latex("") == ""
    
    def test_escape_deeply_nested(self):
        """Test escaping of deeply nested structures."""
        input_data = {
            "level1": {
                "level2": {
                    "level3": [
                        {"text": "Deep & nested"},
                        ["More & text", {"final": "100% complete"}]
                    ]
                }
            }
        }
        expected = {
            "level1": {
                "level2": {
                    "level3": [
                        {"text": "Deep \& nested"},
                        ["More \& text", {"final": "100\% complete"}]
                    ]
                }
            }
        }
        result = recursively_escape_latex(input_data)
        assert result == expected
    
    def test_escape_mixed_types_list(self):
        """Test escaping of list with mixed data types."""
        input_data = [
            "Text with & symbols",
            42,
            {"nested": "More & text"},
            ["Nested list & item"],
            None,
            True
        ]
        expected = [
            "Text with \& symbols",
            42,
            {"nested": "More \& text"},
            ["Nested list \& item"],
            None,
            True
        ]
        result = recursively_escape_latex(input_data)
        assert result == expected
    
    def test_escape_string_input(self):
        """Test that string input is properly escaped."""
        input_text = "Simple & text"
        expected = "Simple \& text"
        result = recursively_escape_latex(input_text)
        assert result == expected
    
    def test_escape_non_container_types(self):
        """Test that non-container types are handled correctly."""
        assert recursively_escape_latex(42) == 42
        assert recursively_escape_latex(3.14) == 3.14
        assert recursively_escape_latex(True) is True
        assert recursively_escape_latex(False) is False
        assert recursively_escape_latex(None) is None


class TestIntegrationScenarios:
    """Integration test scenarios for LaTeX escaping."""
    
    def test_cv_data_structure(self):
        """Test escaping of a realistic CV data structure."""
        cv_data = {
            "personal_info": {
                "name": "John Smith & Associates",
                "email": "john@example.com",
                "phone": "+1-555-123-4567"
            },
            "experience": [
                {
                    "title": "Senior Developer & Team Lead",
                    "company": "Tech Corp (100% Remote)",
                    "achievements": [
                        "Improved performance by 50%",
                        "Led team of 5+ developers",
                        "Worked with C++ & Python"
                    ]
                }
            ],
            "skills": ["Python", "C++", "Data Analysis (90% proficiency)"],
            "metadata": {
                "generated_at": "2024-01-01",
                "version": 1.0,
                "notes": "Top #1 candidate for the role"
            }
        }
        
        result = recursively_escape_latex(cv_data)
        
        # Check that special characters are escaped
        assert result["personal_info"]["name"] == "John Smith \& Associates"
        assert result["experience"][0]["title"] == "Senior Developer \& Team Lead"
        assert result["experience"][0]["company"] == "Tech Corp (100\% Remote)"
        assert "50\%" in result["experience"][0]["achievements"][0]
        assert "5\textasciicircum{}" in result["experience"][0]["achievements"][1]
        assert "C\textasciicircum{}\textasciicircum{} \& Python" in result["experience"][0]["achievements"][2]
        assert "90\% proficiency" in result["skills"][2]
        assert result["metadata"]["notes"] == "Top \#1 candidate for the role"
        
        # Check that non-strings are preserved
        assert result["metadata"]["version"] == 1.0
        assert result["metadata"]["generated_at"] == "2024-01-01"  # No special chars, unchanged
    
    def test_performance_with_large_structure(self):
        """Test performance with a larger data structure."""
        # Create a larger structure
        large_data = {
            "sections": [
                {
                    "name": f"Section {i} & More",
                    "items": [
                        {
                            "title": f"Item {j} with 100% completion",
                            "description": f"Description {j} & details"
                        }
                        for j in range(10)
                    ]
                }
                for i in range(5)
            ]
        }
        
        # This should complete without errors
        result = recursively_escape_latex(large_data)
        
        # Verify some escaping occurred
        assert "\&" in result["sections"][0]["name"]
        assert "100\%" in result["sections"][0]["items"][0]["title"]
        assert "\&" in result["sections"][0]["items"][0]["description"]