---
name: job_description_parser
category: prompt
content_type: job_analysis
description: Prompt for parsing job descriptions into structured fields.
---
**Instructions:**

1. Read the job description carefully.
2. Extract the information for each of the following keys: `skills`, `experience_level`, `responsibilities`, `industry_terms`, `company_values`.
3. Return **ONLY a single, valid JSON object** that strictly adheres to the schema provided below. Do not include any explanatory text, markdown formatting, or anything outside of the JSON object.

**JSON Schema:**

```json
{
  "skills": ["string"],
  "experience_level": "string",
  "responsibilities": ["string"],
  "industry_terms": ["string"],
  "company_values": ["string"]
}
```