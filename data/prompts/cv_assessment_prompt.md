---
name: cv_assessment
category: prompt
content_type: cv_assessment
description: "Analyzes a CV against job requirements and provides a detailed assessment."
---
Analyze the following CV against the job requirements and provide a detailed assessment.

Job Title: {job_title}
Job Requirements: {job_requirements}

CV Summary: {executive_summary}
Experience: {experience}
Skills: {qualifications}

Provide analysis in JSON format with:
1. skill_gaps: List of missing skills that are mentioned in the job requirements
2. strengths: List of strong matching points between CV and job requirements
3. experience_relevance: Score from 0-1 indicating how relevant the experience is
4. keyword_match: Score from 0-1 indicating how well CV keywords match job keywords
5. overall_assessment: Brief summary of the candidate's fit for this role

For each analysis point:
- skill_gaps: Be specific about which required skills are missing
- strengths: Highlight exact matches and transferable skills
- experience_relevance: Consider years of experience, industry relevance, and role similarity
- keyword_match: Compare technical terms, industry jargon, and role-specific language
- overall_assessment: Provide actionable insights for CV improvement

Focus on providing constructive feedback that can guide CV optimization for this specific role.