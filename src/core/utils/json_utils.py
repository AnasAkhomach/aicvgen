"""JSON parsing utilities for LLM responses.

This module provides centralized JSON parsing functionality to eliminate
duplication between LLMCVParserService and ResearchAgent.
"""

import json
import re
from typing import Any, Dict

from src.error_handling.exceptions import LLMResponseParsingError


def parse_llm_json_response(raw_text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response text with robust extraction.

    This function consolidates the JSON parsing logic from LLMCVParserService
    and ResearchAgent to provide a single, reliable way to extract JSON
    from LLM responses.

    Args:
        raw_text: Raw text response from LLM that may contain JSON

    Returns:
        Dict[str, Any]: Parsed JSON data as dictionary

    Raises:
        LLMResponseParsingError: If no valid JSON can be extracted from the text

    Examples:
        >>> parse_llm_json_response('{"key": "value"}')
        {'key': 'value'}

        >>> parse_llm_json_response('```json\n{"key": "value"}\n```')
        {'key': 'value'}
    """
    if not raw_text or not raw_text.strip():
        raise LLMResponseParsingError(
            "Empty or whitespace-only response received from LLM",
            raw_response=raw_text or "",
        )

    # Strategy 1: Try to extract JSON from markdown code blocks
    json_match = re.search(
        r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw_text, re.IGNORECASE
    )
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Continue to next strategy if this fails
            pass

    # Strategy 2: Try to find raw JSON objects or arrays in the text
    # Use a more robust approach to find balanced JSON structures

    # Find all potential JSON object starts
    for i, char in enumerate(raw_text):
        if char == "{":
            # Try to find the matching closing brace
            brace_count = 0
            for j in range(i, len(raw_text)):
                if raw_text[j] == "{":
                    brace_count += 1
                elif raw_text[j] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found a complete JSON object candidate
                        json_candidate = raw_text[i : j + 1].strip()
                        try:
                            return json.loads(json_candidate)
                        except json.JSONDecodeError:
                            # Continue to next potential start
                            break
        elif char == "[":
            # Try to find the matching closing bracket
            bracket_count = 0
            for j in range(i, len(raw_text)):
                if raw_text[j] == "[":
                    bracket_count += 1
                elif raw_text[j] == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        # Found a complete JSON array candidate
                        json_candidate = raw_text[i : j + 1].strip()
                        try:
                            return json.loads(json_candidate)
                        except json.JSONDecodeError:
                            # Continue to next potential start
                            break

    # If all strategies fail, raise an error
    raise LLMResponseParsingError(
        "Could not extract valid JSON from LLM response", raw_response=raw_text
    )
