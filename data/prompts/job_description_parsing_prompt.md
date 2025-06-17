You will extract key information from the job description below:

Your task is to act as an expert job description analyzer. Extract the key information from the following job description text and provide it in a structured JSON format.

**Job Description:**
```
{{raw_text}}
```

**Instructions:**
1.  Read the job description carefully.
2.  Extract the information for each of the following keys: `skills`, `experience_level`, `responsibilities`, `industry_terms`, `company_values`.
3.  Return **ONLY a single, valid JSON object** that strictly adheres to the schema provided below. Do not include any explanatory text, markdown formatting, or anything outside of the JSON object.

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