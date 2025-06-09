"""Mock Responses for E2E Testing.

Provides realistic mock responses for:
- LLM Outputs: "Big 10" skills, experience bullets
- API Errors: Rate limit, timeout, authentication failures
- Vector Search: Relevant CV content matches

Used for consistent and predictable E2E testing scenarios.
"""

import json
from typing import Dict, Any, List
from datetime import datetime


class MockLLMResponses:
    """Mock LLM responses for different scenarios."""
    
    @staticmethod
    def get_big_10_skills_response(job_role: str = "software_engineer") -> Dict[str, Any]:
        """Mock response for Big 10 skills extraction."""
        
        skills_by_role = {
            "software_engineer": [
                "Python Programming",
                "JavaScript/TypeScript",
                "React Framework",
                "Node.js Backend Development",
                "PostgreSQL Database Management",
                "AWS Cloud Services",
                "Docker Containerization",
                "Git Version Control",
                "RESTful API Design",
                "Agile Development Methodologies"
            ],
            "ai_engineer": [
                "Machine Learning Algorithms",
                "Python Programming",
                "TensorFlow/PyTorch",
                "Deep Learning Neural Networks",
                "Natural Language Processing",
                "Computer Vision",
                "MLOps and Model Deployment",
                "AWS/GCP Cloud Platforms",
                "Data Pipeline Engineering",
                "Statistical Analysis"
            ],
            "data_scientist": [
                "Statistical Analysis",
                "Python/R Programming",
                "Machine Learning",
                "Data Visualization",
                "SQL Database Querying",
                "A/B Testing",
                "Tableau/Power BI",
                "Apache Spark",
                "Experimental Design",
                "Business Intelligence"
            ]
        }
        
        skills = skills_by_role.get(job_role, skills_by_role["software_engineer"])
        
        return {
            "content": json.dumps({
                "big_10_skills": skills,
                "skill_categories": {
                    "technical_skills": skills[:6],
                    "tools_and_platforms": skills[6:8],
                    "methodologies": skills[8:]
                },
                "confidence_scores": {skill: 0.85 + (i * 0.02) for i, skill in enumerate(skills)}
            }),
            "tokens_used": 150,
            "model": "gpt-4",
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def get_experience_bullets_response(experience_type: str = "software_development") -> Dict[str, Any]:
        """Mock response for tailored experience bullets."""
        
        bullets_by_type = {
            "software_development": [
                "Developed scalable web applications using React and Node.js, serving 10,000+ daily active users",
                "Implemented RESTful APIs with 99.9% uptime, reducing response times by 40% through optimization",
                "Built automated testing suites achieving 95% code coverage, reducing production bugs by 60%",
                "Collaborated with cross-functional teams in agile environment, delivering features 25% faster",
                "Optimized database queries and implemented caching strategies, improving performance by 50%"
            ],
            "ai_engineering": [
                "Designed and deployed machine learning models achieving 92% accuracy in production environments",
                "Built end-to-end ML pipelines processing 1TB+ daily data using Apache Spark and Kubernetes",
                "Implemented computer vision algorithms reducing manual processing time by 80%",
                "Developed NLP models for sentiment analysis with 89% precision across multiple languages",
                "Established MLOps practices including model versioning, monitoring, and automated retraining"
            ],
            "data_science": [
                "Conducted statistical analysis on 10M+ customer records, identifying $2M revenue opportunities",
                "Built predictive models improving customer retention by 15% through targeted interventions",
                "Designed and executed A/B tests with 95% statistical confidence, optimizing conversion rates",
                "Created interactive dashboards in Tableau, enabling data-driven decisions across 5 departments",
                "Developed time series forecasting models with 85% accuracy for demand planning"
            ]
        }
        
        bullets = bullets_by_type.get(experience_type, bullets_by_type["software_development"])
        
        return {
            "content": json.dumps({
                "tailored_bullets": bullets,
                "impact_metrics": {
                    "quantified_achievements": len([b for b in bullets if any(char.isdigit() for char in b)]),
                    "action_verbs_used": ["Developed", "Implemented", "Built", "Collaborated", "Optimized"],
                    "technical_keywords": ["React", "Node.js", "APIs", "testing", "database"]
                },
                "relevance_score": 0.92
            }),
            "tokens_used": 200,
            "model": "gpt-4",
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def get_project_description_response(project_type: str = "web_application") -> Dict[str, Any]:
        """Mock response for tailored project descriptions."""
        
        projects_by_type = {
            "web_application": {
                "title": "E-commerce Platform Development",
                "description": "Led development of full-stack e-commerce platform using React, Node.js, and PostgreSQL, handling 50,000+ monthly transactions with 99.9% uptime",
                "technologies": ["React", "Node.js", "PostgreSQL", "AWS", "Docker"],
                "achievements": [
                    "Reduced page load times by 60% through performance optimization",
                    "Implemented secure payment processing with PCI compliance",
                    "Built admin dashboard for inventory management and analytics"
                ]
            },
            "machine_learning": {
                "title": "Predictive Analytics Engine",
                "description": "Developed machine learning pipeline for customer behavior prediction using Python, TensorFlow, and Apache Spark, achieving 87% accuracy in production",
                "technologies": ["Python", "TensorFlow", "Apache Spark", "Kubernetes", "MLflow"],
                "achievements": [
                    "Increased customer retention by 20% through targeted recommendations",
                    "Processed 1TB+ daily data with sub-second inference times",
                    "Established automated model retraining and deployment pipeline"
                ]
            },
            "data_analytics": {
                "title": "Business Intelligence Dashboard",
                "description": "Created comprehensive analytics platform using Python, SQL, and Tableau, providing real-time insights for executive decision-making across multiple business units",
                "technologies": ["Python", "SQL", "Tableau", "Apache Airflow", "Snowflake"],
                "achievements": [
                    "Automated 15+ manual reporting processes, saving 40 hours weekly",
                    "Enabled data-driven decisions resulting in 12% revenue increase",
                    "Built self-service analytics tools used by 200+ stakeholders"
                ]
            }
        }
        
        project = projects_by_type.get(project_type, projects_by_type["web_application"])
        
        return {
            "content": json.dumps({
                "tailored_project": project,
                "relevance_indicators": {
                    "technology_match": 0.95,
                    "impact_quantification": 0.88,
                    "business_value": 0.91
                }
            }),
            "tokens_used": 180,
            "model": "gpt-4",
            "timestamp": datetime.now().isoformat()
        }


class MockAPIErrors:
    """Mock API error responses for testing error handling."""
    
    @staticmethod
    def get_rate_limit_error() -> Exception:
        """Mock rate limit error."""
        class RateLimitError(Exception):
            def __init__(self):
                super().__init__("Rate limit exceeded. Please try again in 60 seconds.")
                self.status_code = 429
                self.retry_after = 60
        
        return RateLimitError()
    
    @staticmethod
    def get_timeout_error() -> Exception:
        """Mock timeout error."""
        class TimeoutError(Exception):
            def __init__(self):
                super().__init__("Request timeout after 30 seconds")
                self.status_code = 408
                self.timeout_duration = 30
        
        return TimeoutError()
    
    @staticmethod
    def get_authentication_error() -> Exception:
        """Mock authentication error."""
        class AuthenticationError(Exception):
            def __init__(self):
                super().__init__("Invalid API key or authentication failed")
                self.status_code = 401
                self.error_type = "authentication_failed"
        
        return AuthenticationError()
    
    @staticmethod
    def get_server_error() -> Exception:
        """Mock server error."""
        class ServerError(Exception):
            def __init__(self):
                super().__init__("Internal server error occurred")
                self.status_code = 500
                self.error_type = "internal_server_error"
        
        return ServerError()
    
    @staticmethod
    def get_quota_exceeded_error() -> Exception:
        """Mock quota exceeded error."""
        class QuotaExceededError(Exception):
            def __init__(self):
                super().__init__("Monthly quota exceeded. Please upgrade your plan.")
                self.status_code = 429
                self.error_type = "quota_exceeded"
                self.quota_reset_date = "2024-02-01"
        
        return QuotaExceededError()


class MockVectorSearchResponses:
    """Mock vector search responses for CV content matching."""
    
    @staticmethod
    def get_relevant_experience_matches(query: str = "software development") -> Dict[str, Any]:
        """Mock relevant experience matches from vector search."""
        
        matches_by_query = {
            "software development": [
                {
                    "content": "Developed web applications using React and Node.js",
                    "similarity_score": 0.92,
                    "source_section": "professional_experience",
                    "metadata": {
                        "company": "TechCorp",
                        "role": "Software Engineer",
                        "duration": "2020-2023"
                    }
                },
                {
                    "content": "Built RESTful APIs for mobile applications",
                    "similarity_score": 0.88,
                    "source_section": "professional_experience",
                    "metadata": {
                        "company": "StartupTech",
                        "role": "Full Stack Developer",
                        "duration": "2019-2020"
                    }
                },
                {
                    "content": "Implemented automated testing and CI/CD pipelines",
                    "similarity_score": 0.85,
                    "source_section": "professional_experience",
                    "metadata": {
                        "company": "DevOps Solutions",
                        "role": "DevOps Engineer",
                        "duration": "2018-2019"
                    }
                }
            ],
            "machine learning": [
                {
                    "content": "Developed machine learning models for predictive analytics",
                    "similarity_score": 0.95,
                    "source_section": "professional_experience",
                    "metadata": {
                        "company": "AI Innovations",
                        "role": "ML Engineer",
                        "duration": "2021-2023"
                    }
                },
                {
                    "content": "Implemented deep learning algorithms for computer vision",
                    "similarity_score": 0.91,
                    "source_section": "projects",
                    "metadata": {
                        "project_name": "Image Recognition System",
                        "technologies": ["Python", "TensorFlow", "OpenCV"]
                    }
                }
            ],
            "data science": [
                {
                    "content": "Conducted statistical analysis on large datasets",
                    "similarity_score": 0.93,
                    "source_section": "professional_experience",
                    "metadata": {
                        "company": "DataCorp",
                        "role": "Data Scientist",
                        "duration": "2020-2022"
                    }
                },
                {
                    "content": "Built predictive models for customer segmentation",
                    "similarity_score": 0.89,
                    "source_section": "projects",
                    "metadata": {
                        "project_name": "Customer Analytics Platform",
                        "technologies": ["Python", "Scikit-learn", "Pandas"]
                    }
                }
            ]
        }
        
        matches = matches_by_query.get(query, matches_by_query["software development"])
        
        return {
            "matches": matches,
            "total_results": len(matches),
            "search_metadata": {
                "query": query,
                "search_time_ms": 45,
                "index_size": 1250,
                "algorithm": "cosine_similarity"
            }
        }
    
    @staticmethod
    def get_skill_matches(skills: List[str]) -> Dict[str, Any]:
        """Mock skill matches from CV content."""
        
        skill_matches = {}
        
        for skill in skills:
            if skill.lower() in ["python", "javascript", "react", "node.js"]:
                skill_matches[skill] = {
                    "found_in_cv": True,
                    "confidence": 0.95,
                    "evidence": [
                        f"Mentioned in professional experience: 'Developed applications using {skill}'",
                        f"Listed in technical skills section",
                        f"Referenced in project descriptions"
                    ],
                    "proficiency_level": "advanced"
                }
            elif skill.lower() in ["aws", "docker", "kubernetes", "sql"]:
                skill_matches[skill] = {
                    "found_in_cv": True,
                    "confidence": 0.82,
                    "evidence": [
                        f"Mentioned in project context",
                        f"Listed as supporting technology"
                    ],
                    "proficiency_level": "intermediate"
                }
            else:
                skill_matches[skill] = {
                    "found_in_cv": False,
                    "confidence": 0.0,
                    "evidence": [],
                    "proficiency_level": "none",
                    "suggestion": f"Consider adding {skill} experience or training"
                }
        
        return {
            "skill_matches": skill_matches,
            "overall_match_score": sum(match["confidence"] for match in skill_matches.values()) / len(skills),
            "matched_skills_count": len([s for s in skill_matches.values() if s["found_in_cv"]]),
            "total_skills_analyzed": len(skills)
        }


class MockExpectedOutputs:
    """Expected outputs for validation in E2E tests."""
    
    @staticmethod
    def get_expected_cv_structure() -> Dict[str, Any]:
        """Expected structure of a tailored CV."""
        return {
            "sections": [
                "professional_summary",
                "professional_experience",
                "technical_skills",
                "projects",
                "education",
                "certifications"
            ],
            "required_elements": {
                "big_10_skills": {"min_count": 8, "max_count": 12},
                "experience_bullets": {"min_per_role": 3, "max_per_role": 6},
                "quantified_achievements": {"min_percentage": 60},
                "technical_keywords": {"min_count": 15}
            },
            "formatting_requirements": {
                "max_length_pages": 2,
                "bullet_point_format": True,
                "consistent_tense": True,
                "professional_tone": True
            }
        }
    
    @staticmethod
    def get_expected_processing_metrics() -> Dict[str, Any]:
        """Expected processing performance metrics."""
        return {
            "processing_time": {
                "complete_cv_max_seconds": 120,
                "individual_item_max_seconds": 30,
                "skills_extraction_max_seconds": 15
            },
            "quality_metrics": {
                "min_relevance_score": 0.75,
                "min_content_quality_score": 0.80,
                "max_repetition_percentage": 10
            },
            "rate_limiting": {
                "max_requests_per_minute": 60,
                "max_tokens_per_hour": 100000,
                "retry_backoff_seconds": [1, 2, 4, 8]
            }
        }


def get_mock_response_by_type(response_type: str, **kwargs) -> Dict[str, Any]:
    """Get mock response by type with optional parameters."""
    
    response_generators = {
        "big_10_skills": MockLLMResponses.get_big_10_skills_response,
        "experience_bullets": MockLLMResponses.get_experience_bullets_response,
        "project_description": MockLLMResponses.get_project_description_response,
        "vector_search_experience": MockVectorSearchResponses.get_relevant_experience_matches,
        "vector_search_skills": MockVectorSearchResponses.get_skill_matches,
        "expected_cv_structure": MockExpectedOutputs.get_expected_cv_structure,
        "expected_metrics": MockExpectedOutputs.get_expected_processing_metrics
    }
    
    if response_type not in response_generators:
        raise ValueError(f"Unknown response type: {response_type}. Available: {list(response_generators.keys())}")
    
    generator = response_generators[response_type]
    
    # Handle different parameter signatures
    if response_type == "vector_search_skills":
        return generator(kwargs.get("skills", ["Python", "JavaScript"]))
    elif response_type in ["big_10_skills", "experience_bullets", "project_description", "vector_search_experience"]:
        return generator(kwargs.get("job_role", "software_engineer"))
    else:
        return generator()