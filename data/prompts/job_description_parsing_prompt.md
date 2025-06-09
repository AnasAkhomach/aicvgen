You will extract key information from the job description below:

-----BEGIN JOB DESCRIPTION-----
{raw_text}
-----END JOB DESCRIPTION-----

Extract and return the following in a structured JSON format:
1. A list of specific skills or technologies required (e.g. Python, React, AWS).
2. The experience level required (e.g. Entry-level, Mid-level, Senior, etc.).
3. A list of key responsibilities or tasks from the job.
4. A list of industry-specific terms or keywords.
5. A list of mentions of company values or culture.

Format your response STRICTLY as a JSON object with these keys: skills, experience_level, responsibilities, industry_terms, company_values.

Each value should be:
- skills: Array of strings (specific technologies, programming languages, tools)
- experience_level: String (the required experience level)
- responsibilities: Array of strings (main job duties and responsibilities)
- industry_terms: Array of strings (domain-specific terminology)
- company_values: Array of strings (cultural aspects, values, work environment)

Be thorough but concise. Focus on extracting the most relevant and specific information that would be useful for CV tailoring.