---
name: job_research_analysis
category: prompt
content_type: job_analysis
description: "Analyzes a job description and provides a structured analysis."
---
Analyze the following job description and provide a structured analysis.
Extract the following information:

1. Core technical skills required (list the top 5-7 most important)
2. Soft skills that would be valuable (list the top 3-5)
3. Key performance metrics mentioned or implied
4. Project types the candidate would likely work on
5. Working environment characteristics (team size, collaboration style, etc.)

Format your response as a JSON object with these 5 keys.

Job Description:
{raw_jd}

Additional context:
- Skills already identified: {skills}
- Company: {company_name}
- Position: {job_title}

Provide a thorough analysis that will help tailor a CV to match this specific role. Focus on:
- Technical requirements that are must-haves vs nice-to-haves
- Soft skills that align with the company culture
- Quantifiable achievements that would be most impressive
- Types of projects that demonstrate relevant experience
- Team dynamics and collaboration expectations

Return your analysis as a well-structured JSON object with the specified keys.