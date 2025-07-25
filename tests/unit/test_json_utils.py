"""Unit tests for JSON parsing utilities.

Tests the centralized JSON parsing functionality that consolidates
logic from LLMCVParserService and ResearchAgent.
"""

import json
import pytest
from unittest.mock import patch

from src.utils.json_utils import parse_llm_json_response
from src.error_handling.exceptions import LLMResponseParsingError


class TestParseLLMJsonResponse:
    """Test cases for parse_llm_json_response function."""

    def test_parse_simple_json_object(self):
        """Test parsing a simple JSON object."""
        raw_text = '{"key": "value", "number": 42}'
        result = parse_llm_json_response(raw_text)
        expected = {"key": "value", "number": 42}
        assert result == expected

    def test_parse_simple_json_array(self):
        """Test parsing a simple JSON array."""
        raw_text = '[{"name": "John"}, {"name": "Jane"}]'
        result = parse_llm_json_response(raw_text)
        expected = [{"name": "John"}, {"name": "Jane"}]
        assert result == expected

    def test_parse_json_from_markdown_code_block(self):
        """Test extracting JSON from markdown code blocks."""
        raw_text = '''Here is the JSON data:
```json
{"status": "success", "data": [1, 2, 3]}
```
That's the result.'''
        result = parse_llm_json_response(raw_text)
        expected = {"status": "success", "data": [1, 2, 3]}
        assert result == expected

    def test_parse_json_from_code_block_without_language(self):
        """Test extracting JSON from code blocks without language specification."""
        raw_text = '''Response:
```
{"message": "hello world"}
```'''
        result = parse_llm_json_response(raw_text)
        expected = {"message": "hello world"}
        assert result == expected

    def test_parse_json_mixed_with_text(self):
        """Test extracting JSON when mixed with other text."""
        raw_text = '''The analysis shows that {"confidence": 0.95, "category": "positive"} based on the input.'''
        result = parse_llm_json_response(raw_text)
        expected = {"confidence": 0.95, "category": "positive"}
        assert result == expected

    def test_parse_nested_json_object(self):
        """Test parsing complex nested JSON structures."""
        raw_text = '''{
  "user": {
    "name": "Alice",
    "details": {
      "age": 30,
      "skills": ["Python", "JavaScript"]
    }
  },
  "timestamp": "2024-01-01T00:00:00Z"
}'''
        result = parse_llm_json_response(raw_text)
        expected = {
            "user": {
                "name": "Alice",
                "details": {
                    "age": 30,
                    "skills": ["Python", "JavaScript"]
                }
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
        assert result == expected

    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with extra whitespace."""
        raw_text = '''   

   {"clean": true}   

   '''
        result = parse_llm_json_response(raw_text)
        expected = {"clean": True}
        assert result == expected

    def test_parse_first_valid_json_when_multiple_present(self):
        """Test that the first valid JSON is returned when multiple are present."""
        raw_text = '''First: {"first": 1} and second: {"second": 2}'''
        result = parse_llm_json_response(raw_text)
        expected = {"first": 1}
        assert result == expected

    def test_fallback_to_raw_json_when_markdown_fails(self):
        """Test fallback to raw JSON extraction when markdown parsing fails."""
        raw_text = '''```json
{invalid json}
```
But here's valid JSON: {"valid": true}'''
        result = parse_llm_json_response(raw_text)
        expected = {"valid": True}
        assert result == expected

    def test_parse_json_array_in_text(self):
        """Test parsing JSON arrays embedded in text."""
        raw_text = '''The results are: ["item1", "item2", "item3"] as shown.'''
        result = parse_llm_json_response(raw_text)
        expected = ["item1", "item2", "item3"]
        assert result == expected

    def test_empty_string_raises_error(self):
        """Test that empty string raises LLMResponseParsingError."""
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response("")
        assert "Empty or whitespace-only response" in str(exc_info.value)
        assert exc_info.value.context.additional_data['raw_response'] == ""

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only string raises LLMResponseParsingError."""
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response("   \n\t   ")
        assert "Empty or whitespace-only response" in str(exc_info.value)

    def test_none_input_raises_error(self):
        """Test that None input raises LLMResponseParsingError."""
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response(None)
        assert "Empty or whitespace-only response" in str(exc_info.value)
        assert exc_info.value.context.additional_data['raw_response'] == ""

    def test_no_json_content_raises_error(self):
        """Test that text without JSON raises LLMResponseParsingError."""
        raw_text = "This is just plain text without any JSON content."
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response(raw_text)
        assert "Could not extract valid JSON" in str(exc_info.value)
        assert exc_info.value.context.additional_data['raw_response'] == raw_text

    def test_invalid_json_syntax_raises_error(self):
        """Test that invalid JSON syntax raises LLMResponseParsingError."""
        raw_text = '{"key": value, "missing_quotes": true}'
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response(raw_text)
        assert "Could not extract valid JSON" in str(exc_info.value)

    def test_malformed_markdown_with_invalid_json_raises_error(self):
        """Test that malformed JSON in markdown raises error."""
        raw_text = '''```json
{"key": value missing quotes}
```'''
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response(raw_text)
        assert "Could not extract valid JSON" in str(exc_info.value)

    def test_partial_json_objects_raise_error(self):
        """Test that partial/incomplete JSON objects raise error."""
        raw_text = '{"incomplete": '
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response(raw_text)
        assert "Could not extract valid JSON" in str(exc_info.value)

    def test_json_with_trailing_comma_raises_error(self):
        """Test that JSON with trailing comma raises error."""
        raw_text = '{"key": "value",}'
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response(raw_text)
        assert "Could not extract valid JSON" in str(exc_info.value)

    def test_error_contains_raw_response_context(self):
        """Test that errors include the raw response in context."""
        raw_text = "Invalid content"
        with pytest.raises(LLMResponseParsingError) as exc_info:
            parse_llm_json_response(raw_text)
        
        error = exc_info.value
        assert error.context.additional_data['raw_response'] == raw_text
        assert error.context.additional_data['response_length'] == len(raw_text)

    def test_case_insensitive_markdown_detection(self):
        """Test that markdown code block detection is case insensitive."""
        raw_text = '''```JSON
{"case": "insensitive"}
```'''
        result = parse_llm_json_response(raw_text)
        expected = {"case": "insensitive"}
        assert result == expected

    def test_multiple_markdown_blocks_uses_first_valid(self):
        """Test that when multiple markdown blocks exist, first valid one is used."""
        raw_text = '''```json
{invalid}
```

```json
{"valid": "second"}
```'''
        result = parse_llm_json_response(raw_text)
        expected = {"valid": "second"}
        assert result == expected

    def test_json_with_unicode_characters(self):
        """Test parsing JSON with unicode characters."""
        raw_text = '{"message": "Hello 世界", "emoji": "🚀"}'
        result = parse_llm_json_response(raw_text)
        expected = {"message": "Hello 世界", "emoji": "🚀"}
        assert result == expected

    def test_json_with_escaped_characters(self):
        """Test parsing JSON with escaped characters."""
        raw_text = '{"path": "C:\\\\Users\\\\file.txt", "quote": "He said \\"hello\\""}'
        result = parse_llm_json_response(raw_text)
        expected = {"path": "C:\\Users\\file.txt", "quote": 'He said "hello"'}
        assert result == expected

    def test_large_json_object(self):
        """Test parsing a large JSON object."""
        large_data = {f"key_{i}": f"value_{i}" for i in range(100)}
        raw_text = json.dumps(large_data)
        result = parse_llm_json_response(raw_text)
        assert result == large_data
        assert len(result) == 100