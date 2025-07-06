---
name: job_research_analysis
category: prompt
content_type: job_analysis
description: "Analyzes a job description and provides a structured analysis."
---
<instructions>
Your task is to analyze the provided job description and return a single, valid JSON object.
The JSON object must be enclosed in a ```json markdown code block.
The JSON object must contain exactly these 5 keys: "core_technical_skills", "soft_skills", "key_performance_metrics", "project_types", "working_environment_characteristics".
If you cannot find information for a specific key, return an empty array [] as its value. Do not omit any keys.
Each array should contain strings, not objects.
</instructions>

<example>
```json
{{
  "core_technical_skills": ["Python", "JavaScript", "React", "Node.js", "PostgreSQL", "AWS", "Docker"],
  "soft_skills": ["Team collaboration", "Problem solving", "Communication", "Adaptability"],
  "key_performance_metrics": ["Code quality metrics", "Feature delivery timelines", "Bug resolution time", "User satisfaction scores"],
  "project_types": ["Web application development", "API integration", "Database optimization", "Cloud migration"],
  "working_environment_characteristics": ["Agile methodology", "Remote-friendly", "Cross-functional teams", "Continuous integration"]
}}
```
</example>

<input_data>
Job Title: {job_title}
Company: {company_name}
Skills Already Identified: {skills}

Job Description:
{raw_jd}
</input_data>

<analysis_guidelines>
Focus your analysis on:
- Technical requirements that are must-haves vs nice-to-haves
- Soft skills that align with the company culture
- Quantifiable achievements that would be most impressive
- Types of projects that demonstrate relevant experience
- Team dynamics and collaboration expectations
</analysis_guidelines>

Return ONLY the JSON object enclosed in a ```json code block. Do not include any other text or explanations.