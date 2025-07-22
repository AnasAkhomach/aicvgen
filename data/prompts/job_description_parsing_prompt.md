---
name: job_description_parser
category: prompt
content_type: JOB_ANALYSIS
description: Prompt for parsing job descriptions into structured fields.
---
**Instructions:**

1. Read the job description carefully.
2. Extract the information for each of the following keys: `job_title`, `company_name`, `main_job_description_raw`, `skills`, `experience_level`, `responsibilities`, `industry_terms`, `company_values`.
3. Return **ONLY a single, valid JSON object** that strictly adheres to the schema provided below. Do not include any explanatory text, markdown formatting, or anything outside of the JSON object.

**Job Description:**
{raw_job_description}

**JSON Schema:**

```
{{
  "job_title": "string",
  "company_name": "string", 
  "main_job_description_raw": "string",
  "skills": ["string"],
  "experience_level": "string",
  "responsibilities": ["string"],
  "industry_terms": ["string"],
  "company_values": ["string"]
}}
```