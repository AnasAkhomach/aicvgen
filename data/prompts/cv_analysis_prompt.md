---
name: cv_analysis
category: prompt
content_type: cv_analysis
description: "Analyzes CV text and extracts key sections and information into a JSON format."
---
Analyze the following CV text and extract the key sections and information.
Provide the output in JSON format with the following keys:
"summary": The professional summary or objective statement.
"experiences": A list of work experiences, each as a string describing the role, company, and key achievements.
"skills": A list of technical and soft skills mentioned.
"education": A list of educational qualifications (degrees, institutions, dates).
"projects": A list of significant projects mentioned. IMPORTANT: Be thorough in extracting all projects, including project names, technologies used, and key accomplishments.

If a section is not present, provide an empty string or an empty list accordingly.
Pay special attention to the projects section, as it is critical information for the CV.

Job Description Context (for relevance):
{job_description}

CV Text:
{raw_cv_text}

Please analyze the CV thoroughly and extract all relevant information in the specified JSON format.