import re


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON content from LLM response, handling common formatting issues.
    Utility function for parsing JSON from LLM responses.

    Args:
        response: Raw response from LLM

    Returns:
        str: Cleaned JSON string
    """
    # Remove markdown code blocks
    response = re.sub(r"```(?:json)?\s*", "", response)
    response = re.sub(r"```\s*$", "", response)

    # Remove leading/trailing whitespace
    response = response.strip()

    # Try to find JSON object boundaries with proper bracket counting
    # For now, just return the cleaned string
    return response
