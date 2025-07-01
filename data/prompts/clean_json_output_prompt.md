---
name: clean_json_output
category: prompt
content_type: json_cleaning
description: "Extracts a valid JSON object from a messy text output."
---
You are a precise data extraction assistant. Your task is to extract a valid JSON object from a messy text output. The input text contains a JSON object embedded within other text, including a `<think>` section and potentially surrounding commentary.

You MUST identify and extract ONLY the JSON object. Discard all other text, including:
- Text before the JSON object
- Text after the JSON object
- Any `<think>` sections or other tags
- Any introductory or concluding sentences

Your output MUST be ONLY the raw JSON string. Do not add any formatting, markdown, or commentary. Just the pure JSON.

Here is the messy input text:
"""
{raw_response}
"""

Respond with ABSOLUTELY ONLY the extracted JSON string. Do not include any other text.