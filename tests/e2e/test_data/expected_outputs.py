"""Expected outputs and quality metrics for E2E testing validation."""

from typing import Dict, Any, List, Optional
import re


class ExpectedCVOutputs:
    """Expected CV outputs for different job roles and sections."""
    
    @staticmethod
    def get_software_engineer_outputs() -> Dict[str, Any]:
        return {
            "executive_summary": {
                "required_keywords": ["software engineer", "backend", "python", "api", "scalable"],
                "min_length": 100,
                "max_length": 300,
                "tone": "professional",
                "structure": "paragraph"
            },
            "professional_experience": {
                "required_keywords": ["developed", "implemented", "designed", "optimized"],
                "min_bullets": 3,
                "max_bullets": 7,
                "quantified_achievements": True,
                "action_verbs": True
            },
            "technical_skills": {
                "required_skills": ["Python", "REST API", "PostgreSQL", "Docker", "AWS"],
                "categorized": True,
                "relevant_to_job": True
            },
            "projects": {
                "min_projects": 2,
                "max_projects": 4,
                "technical_details": True,
                "impact_metrics": True
            }
        }
    
    @staticmethod
    def get_ai_engineer_outputs() -> Dict[str, Any]:
        return {
            "executive_summary": {
                "required_keywords": ["ai", "machine learning", "python", "ml", "models"],
                "min_length": 100,
                "max_length": 300,
                "tone": "professional",
                "structure": "paragraph"
            },
            "professional_experience": {
                "required_keywords": ["developed", "implemented", "trained", "deployed"],
                "min_bullets": 3,
                "max_bullets": 7,
                "quantified_achievements": True,
                "action_verbs": True
            },
            "technical_skills": {
                "required_skills": ["Python", "TensorFlow", "PyTorch", "MLOps", "AWS"],
                "categorized": True,
                "relevant_to_job": True
            },
            "projects": {
                "min_projects": 2,
                "max_projects": 4,
                "technical_details": True,
                "impact_metrics": True
            }
        }
    
    @staticmethod
    def get_data_scientist_outputs() -> Dict[str, Any]:
        return {
            "executive_summary": {
                "required_keywords": ["data scientist", "analytics", "python", "statistical", "insights"],
                "min_length": 100,
                "max_length": 300,
                "tone": "professional",
                "structure": "paragraph"
            },
            "professional_experience": {
                "required_keywords": ["analyzed", "developed", "created", "implemented"],
                "min_bullets": 3,
                "max_bullets": 7,
                "quantified_achievements": True,
                "action_verbs": True
            },
            "technical_skills": {
                "required_skills": ["Python", "R", "pandas", "scikit-learn", "SQL"],
                "categorized": True,
                "relevant_to_job": True
            },
            "projects": {
                "min_projects": 2,
                "max_projects": 4,
                "technical_details": True,
                "impact_metrics": True
            }
        }


class CVQualityMetrics:
    """Quality metrics and validation criteria for CV content."""
    
    @staticmethod
    def get_content_quality_criteria() -> Dict[str, Any]:
        return {
            "relevance": {
                "weight": 0.3,
                "min_score": 0.8,
                "description": "Content relevance to job requirements"
            },
            "clarity": {
                "weight": 0.25,
                "min_score": 0.8,
                "description": "Clarity and readability of content"
            },
            "completeness": {
                "weight": 0.25,
                "min_score": 0.8,
                "description": "Completeness of required sections"
            },
            "impact": {
                "weight": 0.2,
                "min_score": 0.7,
                "description": "Impact and achievement quantification"
            }
        }
    
    @staticmethod
    def get_formatting_standards() -> Dict[str, Any]:
        return {
            "consistency": {
                "bullet_points": True,
                "date_format": True,
                "section_headers": True
            },
            "length": {
                "max_pages": 2,
                "section_balance": True
            },
            "readability": {
                "clear_hierarchy": True,
                "appropriate_spacing": True,
                "professional_tone": True
            }
        }
    
    @staticmethod
    def get_ats_compatibility_requirements() -> Dict[str, Any]:
        return {
            "keyword_optimization": {
                "job_relevant_keywords": True,
                "keyword_density": {"min": 0.02, "max": 0.08}
            },
            "structure": {
                "clear_sections": True,
                "standard_headings": True,
                "chronological_order": True
            },
            "formatting": {
                "simple_formatting": True,
                "no_complex_layouts": True,
                "readable_fonts": True
            }
        }


def get_expected_output_by_section(section_name: str, job_role: str) -> Dict[str, Any]:
    """Get expected output criteria for a specific section and job role."""
    role_outputs = {
        "software_engineer": ExpectedCVOutputs.get_software_engineer_outputs(),
        "ai_engineer": ExpectedCVOutputs.get_ai_engineer_outputs(),
        "data_scientist": ExpectedCVOutputs.get_data_scientist_outputs()
    }
    
    if job_role not in role_outputs:
        raise ValueError(f"Unknown job role: {job_role}")
    
    if section_name not in role_outputs[job_role]:
        raise ValueError(f"Unknown section: {section_name} for role: {job_role}")
    
    return role_outputs[job_role][section_name]


def validate_cv_section_quality(structured_cv_or_content, expected_cv_output_or_criteria, job_role=None):
    """
    Validate CV section quality. Can be called in two ways:
    1. validate_cv_section_quality(section_content: str, expected_criteria: Dict) -> Dict
    2. validate_cv_section_quality(structured_cv, expected_cv_output: Dict, job_role: str) -> CVQualityMetrics
    """
    # If called with 3 parameters (structured_cv, expected_cv_output, job_role)
    if job_role is not None:
        return _validate_full_cv_quality(structured_cv_or_content, expected_cv_output_or_criteria, job_role)
    
    # If called with 2 parameters (section_content, expected_criteria) - original function
    return _validate_single_section_quality(structured_cv_or_content, expected_cv_output_or_criteria)


def _validate_single_section_quality(section_content: str, expected_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Validate CV section content against expected criteria."""
    validation_result = {
        "passed": True,
        "score": 0.0,
        "issues": [],
        "details": {}
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    # Validate required keywords
    if "required_keywords" in expected_criteria:
        keywords = expected_criteria["required_keywords"]
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in section_content.lower():
                found_keywords.append(keyword)
        
        keyword_score = len(found_keywords) / len(keywords) if keywords else 1.0
        validation_result["details"]["keyword_score"] = keyword_score
        validation_result["details"]["found_keywords"] = found_keywords
        
        if keyword_score < 0.6:  # At least 60% of keywords should be present
            validation_result["issues"].append(f"Missing required keywords: {set(keywords) - set(found_keywords)}")
            validation_result["passed"] = False
        
        total_score += keyword_score * 0.3
        total_weight += 0.3
    
    # Validate length requirements
    if "min_length" in expected_criteria or "max_length" in expected_criteria:
        content_length = len(section_content.strip())
        length_valid = True
        
        if "min_length" in expected_criteria and content_length < expected_criteria["min_length"]:
            validation_result["issues"].append(f"Content too short: {content_length} < {expected_criteria['min_length']}")
            length_valid = False
        
        if "max_length" in expected_criteria and content_length > expected_criteria["max_length"]:
            validation_result["issues"].append(f"Content too long: {content_length} > {expected_criteria['max_length']}")
            length_valid = False
        
        if not length_valid:
            validation_result["passed"] = False
        
        length_score = 1.0 if length_valid else 0.5
        validation_result["details"]["length_score"] = length_score
        validation_result["details"]["content_length"] = content_length
        
        total_score += length_score * 0.2
        total_weight += 0.2
    
    # Validate bullet points (for experience sections)
    if "min_bullets" in expected_criteria or "max_bullets" in expected_criteria:
        bullet_pattern = r'^\s*[•\-\*]\s+'
        bullets = re.findall(bullet_pattern, section_content, re.MULTILINE)
        bullet_count = len(bullets)
        
        bullets_valid = True
        if "min_bullets" in expected_criteria and bullet_count < expected_criteria["min_bullets"]:
            validation_result["issues"].append(f"Too few bullet points: {bullet_count} < {expected_criteria['min_bullets']}")
            bullets_valid = False
        
        if "max_bullets" in expected_criteria and bullet_count > expected_criteria["max_bullets"]:
            validation_result["issues"].append(f"Too many bullet points: {bullet_count} > {expected_criteria['max_bullets']}")
            bullets_valid = False
        
        if not bullets_valid:
            validation_result["passed"] = False
        
        bullet_score = 1.0 if bullets_valid else 0.5
        validation_result["details"]["bullet_score"] = bullet_score
        validation_result["details"]["bullet_count"] = bullet_count
        
        total_score += bullet_score * 0.2
        total_weight += 0.2
    
    # Validate action verbs (for experience sections)
    if expected_criteria.get("action_verbs", False):
        action_verbs = [
            "developed", "implemented", "designed", "created", "built", "led", "managed",
            "optimized", "improved", "achieved", "delivered", "collaborated", "analyzed"
        ]
        
        found_verbs = []
        for verb in action_verbs:
            if verb.lower() in section_content.lower():
                found_verbs.append(verb)
        
        verb_score = min(len(found_verbs) / 3, 1.0)  # At least 3 action verbs expected
        validation_result["details"]["action_verb_score"] = verb_score
        validation_result["details"]["found_action_verbs"] = found_verbs
        
        if verb_score < 0.5:
            validation_result["issues"].append("Insufficient use of strong action verbs")
            validation_result["passed"] = False
        
        total_score += verb_score * 0.15
        total_weight += 0.15
    
    # Validate quantified achievements
    if expected_criteria.get("quantified_achievements", False):
        number_pattern = r'\d+[%\+]?|\$\d+|\d+[kmb]\+?'
        numbers = re.findall(number_pattern, section_content, re.IGNORECASE)
        
        quantification_score = min(len(numbers) / 2, 1.0)  # At least 2 quantified achievements
        validation_result["details"]["quantification_score"] = quantification_score
        validation_result["details"]["found_metrics"] = numbers
        
        if quantification_score < 0.5:
            validation_result["issues"].append("Insufficient quantified achievements")
        
        total_score += quantification_score * 0.15
        total_weight += 0.15
    
    # Calculate final score
    if total_weight > 0:
        validation_result["score"] = total_score / total_weight
    else:
        validation_result["score"] = 1.0  # Default score if no criteria to validate
    
    # Overall pass/fail based on score
    if validation_result["score"] < 0.7:
        validation_result["passed"] = False
    
    return validation_result


def _validate_full_cv_quality(structured_cv, expected_cv_output: Dict[str, Any], job_role: str):
    """Validate full CV quality and return CVQualityMetrics object."""
    from types import SimpleNamespace
    
    # Get expected outputs for the job role
    role_outputs = expected_cv_output.get(job_role, {})
    
    # Initialize scores
    overall_scores = []
    content_relevance_scores = []
    keyword_alignment_scores = []
    
    # Validate each section of the CV
    cv_sections = {
        'executive_summary': getattr(structured_cv, 'executive_summary', ''),
        'experience': '\n'.join([f"{exp.company} - {exp.role}\n{exp.description}" 
                                for exp in getattr(structured_cv, 'experience', [])]),
        'education': '\n'.join([f"{edu.institution} - {edu.degree}" 
                               for edu in getattr(structured_cv, 'education', [])]),
        'skills': ', '.join(getattr(structured_cv, 'skills', [])),
        'big_10_skills': ', '.join(getattr(structured_cv, 'big_10_skills', []))
    }
    
    for section_name, section_content in cv_sections.items():
        if section_name in role_outputs and section_content:
            expected_criteria = role_outputs[section_name]
            section_result = _validate_single_section_quality(section_content, expected_criteria)
            
            overall_scores.append(section_result['score'])
            
            # Content relevance based on keyword score
            keyword_score = section_result.get('details', {}).get('keyword_score', 0.8)
            content_relevance_scores.append(keyword_score)
            
            # Keyword alignment is the same as keyword score
            keyword_alignment_scores.append(keyword_score)
    
    # Calculate aggregate scores
    overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.8
    content_relevance = sum(content_relevance_scores) / len(content_relevance_scores) if content_relevance_scores else 0.8
    keyword_alignment = sum(keyword_alignment_scores) / len(keyword_alignment_scores) if keyword_alignment_scores else 0.8
    
    # Create and return CVQualityMetrics-like object
    quality_metrics = SimpleNamespace()
    quality_metrics.overall_score = overall_score
    quality_metrics.content_relevance = content_relevance
    quality_metrics.keyword_alignment = keyword_alignment
    
    return quality_metrics