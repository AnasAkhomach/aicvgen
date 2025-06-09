"""Expected Outputs for E2E Test Validation.

Provides pre-validated CV sections and expected results for comparison
during end-to-end testing. Used to ensure consistent quality and
accuracy of CV tailoring outputs.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ExpectedCVSection:
    """Expected CV section with validation criteria."""
    section_name: str
    required_elements: List[str]
    quality_criteria: Dict[str, Any]
    sample_content: str
    validation_rules: Dict[str, Any]


class ExpectedCVOutputs:
    """Expected outputs for different CV tailoring scenarios."""
    
    @staticmethod
    def get_expected_professional_summary(job_role: str = "software_engineer") -> ExpectedCVSection:
        """Expected professional summary section."""
        
        summaries_by_role = {
            "software_engineer": {
                "sample_content": """
                Experienced software engineer with 5+ years of expertise in full-stack web development. 
                Proven track record of building scalable applications using React, Node.js, and cloud technologies. 
                Strong background in agile methodologies, test-driven development, and DevOps practices. 
                Passionate about creating efficient, maintainable code and mentoring junior developers.
                """,
                "required_elements": [
                    "years_of_experience",
                    "key_technologies",
                    "notable_achievements",
                    "professional_strengths"
                ],
                "quality_criteria": {
                    "word_count_range": (50, 100),
                    "technical_keywords_min": 5,
                    "quantified_experience": True,
                    "action_oriented_language": True
                }
            },
            "ai_engineer": {
                "sample_content": """
                Senior AI Engineer with 7+ years of experience designing and deploying machine learning systems at scale. 
                Expertise in deep learning, NLP, and computer vision using TensorFlow and PyTorch. 
                Proven ability to translate research into production-ready AI solutions serving millions of users. 
                Strong background in MLOps, cloud architecture, and cross-functional collaboration.
                """,
                "required_elements": [
                    "ai_ml_experience",
                    "technical_frameworks",
                    "scale_indicators",
                    "research_to_production"
                ],
                "quality_criteria": {
                    "word_count_range": (60, 120),
                    "ai_keywords_min": 8,
                    "quantified_impact": True,
                    "technical_depth": True
                }
            },
            "data_scientist": {
                "sample_content": """
                Data Scientist with 6+ years of experience driving business insights through advanced analytics and machine learning. 
                Expert in statistical modeling, A/B testing, and predictive analytics using Python and R. 
                Proven track record of delivering data-driven solutions that increased revenue by 20%+ and improved operational efficiency. 
                Strong communicator with ability to translate complex findings into actionable business recommendations.
                """,
                "required_elements": [
                    "analytics_experience",
                    "statistical_methods",
                    "business_impact",
                    "communication_skills"
                ],
                "quality_criteria": {
                    "word_count_range": (70, 130),
                    "business_metrics": True,
                    "statistical_keywords_min": 6,
                    "impact_quantification": True
                }
            }
        }
        
        role_data = summaries_by_role.get(job_role, summaries_by_role["software_engineer"])
        
        return ExpectedCVSection(
            section_name="professional_summary",
            required_elements=role_data["required_elements"],
            quality_criteria=role_data["quality_criteria"],
            sample_content=role_data["sample_content"].strip(),
            validation_rules={
                "max_sentences": 4,
                "min_sentences": 2,
                "professional_tone": True,
                "first_person_avoided": True,
                "present_tense_for_current": True
            }
        )
    
    @staticmethod
    def get_expected_experience_bullets(job_role: str = "software_engineer") -> ExpectedCVSection:
        """Expected experience bullets section."""
        
        bullets_by_role = {
            "software_engineer": {
                "sample_content": """
                • Developed scalable web applications using React and Node.js, serving 50,000+ daily active users
                • Implemented RESTful APIs with 99.9% uptime, reducing response times by 40% through optimization
                • Built comprehensive testing suites achieving 95% code coverage, reducing production bugs by 60%
                • Led agile development team of 5 engineers, delivering features 25% faster than previous quarters
                • Optimized database queries and caching strategies, improving application performance by 50%
                """,
                "required_elements": [
                    "action_verbs",
                    "quantified_results",
                    "technical_technologies",
                    "business_impact",
                    "team_collaboration"
                ],
                "quality_criteria": {
                    "bullets_per_role_range": (3, 6),
                    "quantified_achievements_percentage": 80,
                    "action_verb_variety": 5,
                    "technical_keywords_per_bullet": 2
                }
            },
            "ai_engineer": {
                "sample_content": """
                • Designed and deployed machine learning models achieving 92% accuracy in production environments
                • Built end-to-end ML pipelines processing 1TB+ daily data using Apache Spark and Kubernetes
                • Implemented computer vision algorithms reducing manual processing time by 80%
                • Developed NLP models for sentiment analysis with 89% precision across 15 languages
                • Established MLOps practices including automated model retraining, reducing deployment time by 70%
                """,
                "required_elements": [
                    "ml_algorithms",
                    "model_performance",
                    "data_scale",
                    "automation_impact",
                    "technical_frameworks"
                ],
                "quality_criteria": {
                    "bullets_per_role_range": (4, 7),
                    "accuracy_metrics_percentage": 60,
                    "ml_keywords_per_bullet": 3,
                    "scale_indicators": True
                }
            },
            "data_scientist": {
                "sample_content": """
                • Conducted statistical analysis on 10M+ customer records, identifying $2M revenue opportunities
                • Built predictive models improving customer retention by 15% through targeted interventions
                • Designed and executed A/B tests with 95% statistical confidence, optimizing conversion rates by 12%
                • Created interactive dashboards in Tableau, enabling data-driven decisions across 5 departments
                • Developed time series forecasting models with 85% accuracy for demand planning and inventory optimization
                """,
                "required_elements": [
                    "statistical_methods",
                    "business_metrics",
                    "data_volume",
                    "visualization_tools",
                    "predictive_modeling"
                ],
                "quality_criteria": {
                    "bullets_per_role_range": (4, 6),
                    "business_impact_percentage": 90,
                    "statistical_confidence": True,
                    "data_scale_indicators": True
                }
            }
        }
        
        role_data = bullets_by_role.get(job_role, bullets_by_role["software_engineer"])
        
        return ExpectedCVSection(
            section_name="professional_experience",
            required_elements=role_data["required_elements"],
            quality_criteria=role_data["quality_criteria"],
            sample_content=role_data["sample_content"].strip(),
            validation_rules={
                "bullet_format": True,
                "past_tense_verbs": True,
                "consistent_formatting": True,
                "no_personal_pronouns": True,
                "parallel_structure": True
            }
        )
    
    @staticmethod
    def get_expected_technical_skills(job_role: str = "software_engineer") -> ExpectedCVSection:
        """Expected technical skills section."""
        
        skills_by_role = {
            "software_engineer": {
                "sample_content": """
                **Programming Languages:** Python, JavaScript, TypeScript, Java, SQL
                **Frontend Technologies:** React, Vue.js, HTML5, CSS3, Bootstrap, Material-UI
                **Backend Technologies:** Node.js, Express.js, Django, Flask, Spring Boot
                **Databases:** PostgreSQL, MongoDB, Redis, MySQL, Elasticsearch
                **Cloud & DevOps:** AWS, Docker, Kubernetes, Jenkins, Terraform, Git
                **Testing & Quality:** Jest, Pytest, Selenium, CI/CD, Code Review
                """,
                "required_elements": [
                    "programming_languages",
                    "frameworks_libraries",
                    "databases",
                    "cloud_technologies",
                    "development_tools"
                ],
                "quality_criteria": {
                    "total_skills_range": (15, 30),
                    "categorized_organization": True,
                    "relevance_to_role": 0.85,
                    "current_technologies": True
                }
            },
            "ai_engineer": {
                "sample_content": """
                **Programming Languages:** Python, R, SQL, Scala, Java
                **ML/AI Frameworks:** TensorFlow, PyTorch, Scikit-learn, Keras, XGBoost
                **Data Processing:** Apache Spark, Pandas, NumPy, Dask, Apache Airflow
                **Cloud Platforms:** AWS (SageMaker, EC2, S3), GCP (AI Platform), Azure ML
                **MLOps Tools:** MLflow, Kubeflow, Docker, Kubernetes, Git, DVC
                **Specialized Areas:** NLP, Computer Vision, Deep Learning, Statistical Modeling
                """,
                "required_elements": [
                    "ml_frameworks",
                    "data_processing",
                    "cloud_ml_services",
                    "mlops_tools",
                    "ai_specializations"
                ],
                "quality_criteria": {
                    "total_skills_range": (20, 35),
                    "ml_focus_percentage": 70,
                    "production_tools": True,
                    "research_to_production": True
                }
            },
            "data_scientist": {
                "sample_content": """
                **Programming Languages:** Python, R, SQL, Scala
                **Data Analysis:** Pandas, NumPy, SciPy, Statsmodels, Matplotlib, Seaborn
                **Machine Learning:** Scikit-learn, TensorFlow, XGBoost, Random Forest, SVM
                **Visualization:** Tableau, Power BI, Plotly, D3.js, Jupyter Notebooks
                **Big Data:** Apache Spark, Hadoop, Hive, Apache Kafka, Snowflake
                **Statistical Methods:** A/B Testing, Hypothesis Testing, Regression Analysis, Time Series
                """,
                "required_elements": [
                    "statistical_tools",
                    "visualization_platforms",
                    "big_data_technologies",
                    "ml_algorithms",
                    "analytical_methods"
                ],
                "quality_criteria": {
                    "total_skills_range": (18, 32),
                    "statistical_focus_percentage": 60,
                    "business_intelligence": True,
                    "experimental_design": True
                }
            }
        }
        
        role_data = skills_by_role.get(job_role, skills_by_role["software_engineer"])
        
        return ExpectedCVSection(
            section_name="technical_skills",
            required_elements=role_data["required_elements"],
            quality_criteria=role_data["quality_criteria"],
            sample_content=role_data["sample_content"].strip(),
            validation_rules={
                "categorized_format": True,
                "bold_categories": True,
                "comma_separated_items": True,
                "no_skill_repetition": True,
                "alphabetical_within_category": False
            }
        )
    
    @staticmethod
    def get_expected_projects_section(job_role: str = "software_engineer") -> ExpectedCVSection:
        """Expected projects section."""
        
        projects_by_role = {
            "software_engineer": {
                "sample_content": """
                **E-commerce Platform** | *React, Node.js, PostgreSQL, AWS* | *2023*
                • Developed full-stack e-commerce platform handling 10,000+ daily transactions
                • Implemented secure payment processing and inventory management system
                • Achieved 99.9% uptime with automated monitoring and alerting
                
                **Task Management API** | *Python, Django, Redis, Docker* | *2022*
                • Built RESTful API serving 5,000+ concurrent users with sub-200ms response times
                • Designed scalable architecture with caching and database optimization
                • Implemented comprehensive testing suite with 95% code coverage
                """,
                "required_elements": [
                    "project_titles",
                    "technology_stack",
                    "project_timeline",
                    "quantified_impact",
                    "technical_achievements"
                ],
                "quality_criteria": {
                    "projects_count_range": (2, 4),
                    "technologies_per_project": 4,
                    "impact_metrics_percentage": 75,
                    "recent_projects_emphasis": True
                }
            },
            "ai_engineer": {
                "sample_content": """
                **Predictive Analytics Engine** | *Python, TensorFlow, Apache Spark, Kubernetes* | *2023*
                • Developed ML pipeline processing 1TB+ daily data with 87% prediction accuracy
                • Implemented real-time inference serving 10,000+ requests per second
                • Reduced model training time by 60% through distributed computing optimization
                
                **Computer Vision System** | *PyTorch, OpenCV, Docker, AWS SageMaker* | *2022*
                • Built image classification model achieving 94% accuracy on production data
                • Deployed scalable inference pipeline handling 1M+ images daily
                • Implemented automated model retraining with performance monitoring
                """,
                "required_elements": [
                    "ml_model_performance",
                    "data_scale",
                    "inference_capabilities",
                    "automation_features",
                    "production_deployment"
                ],
                "quality_criteria": {
                    "projects_count_range": (2, 3),
                    "ml_frameworks_per_project": 3,
                    "accuracy_metrics": True,
                    "scale_indicators": True
                }
            },
            "data_scientist": {
                "sample_content": """
                **Customer Segmentation Analysis** | *Python, Scikit-learn, Tableau, SQL* | *2023*
                • Analyzed 5M+ customer records to identify high-value segments worth $10M revenue
                • Built clustering models with 82% accuracy for targeted marketing campaigns
                • Created interactive dashboards enabling self-service analytics for 50+ stakeholders
                
                **Demand Forecasting Model** | *R, Time Series, Apache Spark, Snowflake* | *2022*
                • Developed forecasting models improving inventory accuracy by 25%
                • Processed 2 years of historical data across 1,000+ product categories
                • Implemented automated reporting reducing manual analysis time by 80%
                """,
                "required_elements": [
                    "business_problem",
                    "data_volume",
                    "model_accuracy",
                    "business_impact",
                    "stakeholder_value"
                ],
                "quality_criteria": {
                    "projects_count_range": (2, 3),
                    "business_metrics": True,
                    "statistical_rigor": True,
                    "stakeholder_impact": True
                }
            }
        }
        
        role_data = projects_by_role.get(job_role, projects_by_role["software_engineer"])
        
        return ExpectedCVSection(
            section_name="projects",
            required_elements=role_data["required_elements"],
            quality_criteria=role_data["quality_criteria"],
            sample_content=role_data["sample_content"].strip(),
            validation_rules={
                "project_title_format": True,
                "technology_stack_listed": True,
                "timeline_included": True,
                "bullet_point_achievements": True,
                "reverse_chronological": True
            }
        )


class CVQualityMetrics:
    """Quality metrics for CV validation."""
    
    @staticmethod
    def get_content_quality_criteria() -> Dict[str, Any]:
        """Get content quality validation criteria."""
        return {
            "readability": {
                "flesch_reading_ease_min": 60,
                "avg_sentence_length_max": 20,
                "complex_words_percentage_max": 15
            },
            "professional_language": {
                "action_verbs_percentage_min": 70,
                "passive_voice_percentage_max": 10,
                "personal_pronouns_count_max": 0,
                "buzzwords_percentage_max": 5
            },
            "quantification": {
                "quantified_achievements_percentage_min": 60,
                "specific_numbers_count_min": 10,
                "percentage_improvements_count_min": 3,
                "scale_indicators_count_min": 5
            },
            "technical_relevance": {
                "job_keywords_match_percentage_min": 75,
                "technical_skills_coverage_min": 80,
                "industry_terminology_usage": True,
                "current_technology_emphasis": True
            }
        }
    
    @staticmethod
    def get_formatting_standards() -> Dict[str, Any]:
        """Get formatting validation standards."""
        return {
            "structure": {
                "consistent_section_headers": True,
                "logical_section_order": True,
                "appropriate_section_lengths": True,
                "clear_visual_hierarchy": True
            },
            "typography": {
                "consistent_font_usage": True,
                "appropriate_font_sizes": True,
                "proper_spacing": True,
                "professional_formatting": True
            },
            "content_organization": {
                "reverse_chronological_experience": True,
                "grouped_similar_items": True,
                "prioritized_relevant_content": True,
                "eliminated_redundancy": True
            },
            "length_constraints": {
                "total_pages_max": 2,
                "section_balance": True,
                "white_space_optimization": True,
                "content_density_appropriate": True
            }
        }
    
    @staticmethod
    def get_ats_compatibility_requirements() -> Dict[str, Any]:
        """Get ATS (Applicant Tracking System) compatibility requirements."""
        return {
            "parsing_friendly": {
                "standard_section_headers": True,
                "simple_formatting": True,
                "no_complex_layouts": True,
                "machine_readable_text": True
            },
            "keyword_optimization": {
                "job_title_variations": True,
                "skill_keyword_density": True,
                "industry_terminology": True,
                "acronym_expansions": True
            },
            "file_format": {
                "pdf_compatibility": True,
                "text_extractability": True,
                "font_embedding": True,
                "searchable_content": True
            }
        }


def get_expected_output_by_section(section_name: str, job_role: str = "software_engineer") -> ExpectedCVSection:
    """Get expected output for a specific CV section."""
    
    section_generators = {
        "professional_summary": ExpectedCVOutputs.get_expected_professional_summary,
        "professional_experience": ExpectedCVOutputs.get_expected_experience_bullets,
        "technical_skills": ExpectedCVOutputs.get_expected_technical_skills,
        "projects": ExpectedCVOutputs.get_expected_projects_section
    }
    
    if section_name not in section_generators:
        raise ValueError(f"Unknown section: {section_name}. Available: {list(section_generators.keys())}")
    
    return section_generators[section_name](job_role)


def validate_cv_section_quality(section_content: str, expected_section: ExpectedCVSection) -> Dict[str, Any]:
    """Validate CV section against expected quality criteria."""
    
    validation_results = {
        "section_name": expected_section.section_name,
        "passed": True,
        "score": 0.0,
        "issues": [],
        "recommendations": []
    }
    
    # Basic content checks
    word_count = len(section_content.split())
    
    # Check required elements presence
    missing_elements = []
    for element in expected_section.required_elements:
        # Simple keyword-based check (in real implementation, use more sophisticated NLP)
        if element.lower().replace("_", " ") not in section_content.lower():
            missing_elements.append(element)
    
    if missing_elements:
        validation_results["issues"].append(f"Missing required elements: {missing_elements}")
        validation_results["passed"] = False
    
    # Check quality criteria
    quality_score = 1.0
    
    if "word_count_range" in expected_section.quality_criteria:
        min_words, max_words = expected_section.quality_criteria["word_count_range"]
        if not (min_words <= word_count <= max_words):
            validation_results["issues"].append(f"Word count {word_count} outside range {min_words}-{max_words}")
            quality_score -= 0.2
    
    # Check for quantified achievements
    if expected_section.quality_criteria.get("quantified_achievements_percentage"):
        numbers_count = sum(1 for char in section_content if char.isdigit())
        if numbers_count < 3:  # Simplified check
            validation_results["recommendations"].append("Add more quantified achievements with specific numbers")
            quality_score -= 0.1
    
    validation_results["score"] = max(0.0, quality_score)
    
    if validation_results["score"] < 0.7:
        validation_results["passed"] = False
    
    return validation_results