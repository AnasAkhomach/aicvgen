#!/usr/bin/env python3
"""
LaTeX Utility Functions

This module provides utility functions for escaping LaTeX special characters
to prevent compilation errors when generating PDF documents.
"""

import re
from typing import Any


def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters in a string to prevent compilation errors.

    This function replaces LaTeX special characters with their escaped equivalents:
    - & -> \\&
    - % -> \\%
    - $ -> \\$
    - # -> \\#
    - ^ -> \\textasciicircum{}
    - _ -> \\_
    - { -> \\{
    - } -> \\}
    - ~ -> \\textasciitilde{}
    - \\ -> \\textbackslash{}

    Args:
        text: The input string that may contain LaTeX special characters

    Returns:
        str: The escaped string safe for LaTeX compilation

    Example:
        >>> escape_latex("Hello _world_ & Co. 100% #1 {awesome}")
        'Hello \\_world\\_ \\& Co. 100\\% \\#1 \\{awesome\\}'
    """
    if not isinstance(text, str):
        return text

    # Define escape mappings based on web search results
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "+": r"\textasciicircum{}",  # Map + to ^ for programming contexts
        "\\": r"\textbackslash{}",
    }

    # Create regex pattern, sorting by length (longest first) to handle backslash correctly
    regex = re.compile(
        "|".join(
            re.escape(str(key))
            for key in sorted(conv.keys(), key=lambda item: -len(item))
        )
    )

    # Replace using the mapping
    return regex.sub(lambda match: conv[match.group()], text)


def recursively_escape_latex(data: Any) -> Any:
    """
    Recursively escape LaTeX special characters in nested data structures.

    This function traverses dictionaries, lists, and other data structures,
    applying LaTeX escaping only to string values while preserving the
    overall structure and non-string data types.

    Args:
        data: The input data structure (dict, list, str, or other types)

    Returns:
        Any: The data structure with all string values escaped for LaTeX

    Example:
        >>> data = {
        ...     "name": "John & Jane",
        ...     "skills": ["C++", "Data Analysis 100%"],
        ...     "metadata": {"score": 95, "notes": "Top #1 candidate"}
        ... }
        >>> recursively_escape_latex(data)
        {
            "name": "John & Jane",
            "skills": ["C++", "Data Analysis 100%"],
            "metadata": {"score": 95, "notes": "Top #1 candidate"}
        }
    """
    if isinstance(data, dict):
        return {k: recursively_escape_latex(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursively_escape_latex(elem) for elem in data]
    elif isinstance(data, str):
        return escape_latex(data)
    else:
        # For non-string types (int, float, bool, None, etc.), return as-is
        return data
