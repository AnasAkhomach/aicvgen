"""Unit tests for Enhanced Quality Assurance Agent.

Tests content validation, scoring algorithms, quality metrics,
and compliance checking for CV generation."""

import unittest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import tempfile
import os
import sys
from typing import Dict, Any, List
import json

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.models.data_models import (
    ContentType, ProcessingStatus, CVGenerationState, JobDescriptionData,
    ExperienceItem, ProjectItem, QualificationItem
)
from src.services.llm import LLMResponse


class TestEnhancedQualityAssuranceAgent(unittest.TestCase):
    """Test cases for Enhanced Quality Assurance Agent."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_config = Mock()
        self.mock_logger = Mock()
        
        # High-quality CV data for testing
        self.high_quality_cv_data = {
            "personal_info": {
                "name": "John Doe",
                "title": "Senior Software Engineer",
                "email": "john.doe@email.com",
                "phone": "(555) 123-4567",
                "linkedin": "linkedin.com/in/johndoe",
                "location": "San Francisco, CA"
            },
            "summary": "Experienced software engineer with 5+ years in full-stack development, specializing in React and Node.js applications. Proven track record of leading teams and delivering scalable solutions.",
            "experiences": [
                {
                    "title": "Senior Software Engineer",
                    "company": "TechCorp Inc.",
                    "duration": "2021-2023",
                    "responsibilities": [
                        "Developed scalable web applications using React and Node.js, serving 100K+ users",
                        "Led a team of 4 developers on microservices architecture migration",
                        "Improved system performance by 40% through code optimization and caching strategies",
                        "Implemented CI/CD pipelines reducing deployment time by 60%"
                    ],
                    "technologies": ["React", "Node.js", "AWS", "Docker", "Kubernetes"]
                },
                {
                    "title": "Software Engineer",
                    "company": "StartupXYZ",
                    "duration": "2019-2021",
                    "responsibilities": [
                        "Built REST APIs using Python and Django, handling 10K+ requests per minute",
                        "Implemented automated testing suite achieving 95% code coverage",
                        "Collaborated with cross-functional teams to deliver features on time"
                    ],
                    "technologies": ["Python", "Django", "PostgreSQL", "Redis"]
                }
            ],
            "education": [
                {
                    "degree": "Bachelor of Science in Computer Science",
                    "institution": "University ABC",
                    "duration": "2015-2019",
                    "details": "GPA: 3.8/4.0, Magna Cum Laude, Relevant Coursework: Data Structures, Algorithms, Software Engineering"
                }
            ],
            "projects": [
                {
                    "name": "E-commerce Platform",
                    "year": "2022",
                    "description": "Built full-stack e-commerce application with payment integration, inventory management, and real-time analytics",
                    "technologies": ["React", "Node.js", "MongoDB", "Stripe", "AWS"]
                }
            ],
            "skills": {
                "programming_languages": ["Python", "JavaScript", "Java", "TypeScript"],
                "frameworks": ["React", "Node.js", "Django", "Express", "Vue.js"],
                "databases": ["MongoDB", "PostgreSQL", "MySQL", "Redis"],
                "tools": ["Git", "Docker", "Jenkins", "AWS", "Kubernetes"]
            }
        }
        
        # Low-quality CV data for testing
        self.low_quality_cv_data = {
            "personal_info": {
                "name": "John",
                "email": "john@email",  # Invalid email format
                # Missing title, phone, etc.
            },
            "summary": "I am a developer.",  # Too brief
            "experiences": [
                {
                    "title": "Developer",
                    "company": "Company",
                    "duration": "2020-2021",
                    "responsibilities": [
                        "Did coding",  # Vague responsibility
                        "Fixed bugs"   # Not quantified
                    ],
                    "technologies": ["JavaScript"]  # Limited technologies
                }
            ],
            "education": [],  # Missing education
            "projects": [],   # Missing projects
            "skills": {
                "programming_languages": ["JavaScript"]  # Limited skills
            }
        }
        
        # Sample job description for alignment testing
        self.sample_job_data = JobDescriptionData(
            raw_text="Senior Full-Stack Developer position at InnovateTech Solutions",
            company_name="InnovateTech Solutions",
            position_title="Senior Full-Stack Developer",
            required_skills=[
                "JavaScript", "Python", "React", "Node.js", "SQL", "NoSQL",
                "AWS", "Docker", "Microservices", "Leadership"
            ],
            preferred_skills=[
                "Kubernetes", "CI/CD", "Performance Optimization", "Team Leadership"
            ],
            qualifications=["5+ years experience", "Bachelor's degree in Computer Science"],
            responsibilities=[
                "Design and develop web applications",
                "Lead development teams",
                "Optimize system performance",
                "Implement DevOps practices"
            ],
            location="San Francisco, CA"
        )

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_qa_agent_initialization(self, mock_config, mock_logger):
        """Test Quality Assurance Agent initialization."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Verify initialization
        self.assertIsNotNone(qa_agent.llm_service)
        self.assertIsNotNone(qa_agent.settings)
        self.assertIsNotNone(qa_agent.logger)
        self.assertEqual(qa_agent.name, "QualityAssuranceAgent")
        self.assertIn("quality", qa_agent.description.lower())

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_content_validation_comprehensive(self, mock_config, mock_logger):
        """Test comprehensive content validation."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test high-quality content validation
        high_quality_result = qa_agent.validate_content_quality(self.high_quality_cv_data)
        
        self.assertIsInstance(high_quality_result, dict)
        self.assertTrue(high_quality_result.get('is_valid', False))
        self.assertGreater(high_quality_result.get('overall_score', 0), 80)
        self.assertLess(len(high_quality_result.get('issues', [])), 3)
        
        # Test low-quality content validation
        low_quality_result = qa_agent.validate_content_quality(self.low_quality_cv_data)
        
        self.assertIsInstance(low_quality_result, dict)
        self.assertFalse(low_quality_result.get('is_valid', True))
        self.assertLess(low_quality_result.get('overall_score', 100), 60)
        self.assertGreater(len(low_quality_result.get('issues', [])), 3)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_scoring_algorithm_accuracy(self, mock_config, mock_logger):
        """Test accuracy of scoring algorithms."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test individual scoring components
        
        # Personal info scoring
        personal_info_score = qa_agent.score_personal_info(
            self.high_quality_cv_data.personal_info
        )
        self.assertGreater(personal_info_score, 85)
        
        low_personal_info_score = qa_agent.score_personal_info(
            self.low_quality_cv_data.personal_info
        )
        self.assertLess(low_personal_info_score, 60)
        
        # Summary scoring
        summary_score = qa_agent.score_summary(self.high_quality_cv_data.summary)
        self.assertGreater(summary_score, 80)
        
        low_summary_score = qa_agent.score_summary(self.low_quality_cv_data.summary)
        self.assertLess(low_summary_score, 50)
        
        # Experience scoring
        experience_score = qa_agent.score_experiences(self.high_quality_cv_data.experiences)
        self.assertGreater(experience_score, 85)
        
        low_experience_score = qa_agent.score_experiences(self.low_quality_cv_data.experiences)
        self.assertLess(low_experience_score, 60)
        
        # Skills scoring
        skills_score = qa_agent.score_skills(self.high_quality_cv_data.skills)
        self.assertGreater(skills_score, 80)
        
        low_skills_score = qa_agent.score_skills(self.low_quality_cv_data.skills)
        self.assertLess(low_skills_score, 50)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_job_alignment_analysis(self, mock_config, mock_logger):
        """Test job description alignment analysis."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Mock LLM response for alignment analysis
        mock_llm_response = LLMResponse(
            content="""{
                "alignment_score": 92,
                "skill_matches": {
                    "exact_matches": ["JavaScript", "Python", "React", "Node.js", "AWS", "Docker"],
                    "partial_matches": ["Leadership", "Performance Optimization"],
                    "missing_skills": ["Kubernetes"]
                },
                "experience_alignment": {
                    "years_match": true,
                    "role_relevance": 95,
                    "responsibility_overlap": 88
                },
                "education_alignment": {
                    "degree_match": true,
                    "field_relevance": 100
                },
                "recommendations": [
                    "Add Kubernetes experience to skills section",
                    "Emphasize team leadership experience more prominently",
                    "Quantify performance optimization achievements"
                ]
            }""",
            tokens_used=600,
            processing_time=3.0,
            model_used="groq",
            success=True
        )
        
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_llm_response)
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test job alignment analysis
        async def test_alignment():
            alignment_result = await qa_agent.analyze_job_alignment(
                cv_data=self.high_quality_cv_data,
                job_data=self.sample_job_data
            )
            
            # Verify alignment analysis
            self.assertIsInstance(alignment_result, dict)
            self.assertTrue(alignment_result.get('success', False))
            
            alignment_data = alignment_result.get('data')
            self.assertIsNotNone(alignment_data)
            
            # Check alignment score
            alignment_score = alignment_data.get('alignment_score')
            self.assertIsInstance(alignment_score, (int, float))
            self.assertGreaterEqual(alignment_score, 0)
            self.assertLessEqual(alignment_score, 100)
            
            # Check skill matches
            skill_matches = alignment_data.get('skill_matches', {})
            self.assertIn('exact_matches', skill_matches)
            self.assertIn('partial_matches', skill_matches)
            self.assertIn('missing_skills', skill_matches)
            
            # Verify specific matches
            exact_matches = skill_matches.get('exact_matches', [])
            self.assertIn('JavaScript', exact_matches)
            self.assertIn('Python', exact_matches)
            self.assertIn('React', exact_matches)
            
            # Check recommendations
            recommendations = alignment_data.get('recommendations', [])
            self.assertIsInstance(recommendations, list)
            self.assertGreater(len(recommendations), 0)
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_alignment())
        finally:
            loop.close()

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_quality_metrics_calculation(self, mock_config, mock_logger):
        """Test quality metrics calculation."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test comprehensive quality metrics
        quality_metrics = qa_agent.calculate_quality_metrics(self.high_quality_cv_data)
        
        self.assertIsInstance(quality_metrics, dict)
        
        # Check required metrics
        required_metrics = [
            'completeness_score', 'clarity_score', 'relevance_score',
            'formatting_score', 'keyword_density', 'readability_score',
            'quantification_score', 'action_verb_usage'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, quality_metrics)
            self.assertIsInstance(quality_metrics[metric], (int, float))
            self.assertGreaterEqual(quality_metrics[metric], 0)
            self.assertLessEqual(quality_metrics[metric], 100)
        
        # Test specific metric calculations
        
        # Completeness score should be high for complete CV
        self.assertGreater(quality_metrics['completeness_score'], 85)
        
        # Clarity score should be high for well-written content
        self.assertGreater(quality_metrics['clarity_score'], 80)
        
        # Quantification score should be high due to metrics in responsibilities
        self.assertGreater(quality_metrics['quantification_score'], 70)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_compliance_checking(self, mock_config, mock_logger):
        """Test compliance checking for various standards."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test ATS compliance
        ats_compliance = qa_agent.check_ats_compliance(self.high_quality_cv_data)
        
        self.assertIsInstance(ats_compliance, dict)
        self.assertIn('is_compliant', ats_compliance)
        self.assertIn('compliance_score', ats_compliance)
        self.assertIn('issues', ats_compliance)
        self.assertIn('recommendations', ats_compliance)
        
        # High-quality CV should have good ATS compliance
        self.assertTrue(ats_compliance.get('is_compliant', False))
        self.assertGreater(ats_compliance.get('compliance_score', 0), 80)
        
        # Test format compliance
        format_compliance = qa_agent.check_format_compliance(self.high_quality_cv_data)
        
        self.assertIsInstance(format_compliance, dict)
        self.assertIn('format_score', format_compliance)
        self.assertIn('structure_issues', format_compliance)
        
        # Test content guidelines compliance
        content_compliance = qa_agent.check_content_guidelines(self.high_quality_cv_data)
        
        self.assertIsInstance(content_compliance, dict)
        self.assertIn('guideline_score', content_compliance)
        self.assertIn('violations', content_compliance)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_readability_analysis(self, mock_config, mock_logger):
        """Test readability analysis of CV content."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test readability analysis
        readability_result = qa_agent.analyze_readability(self.high_quality_cv_data)
        
        self.assertIsInstance(readability_result, dict)
        
        # Check readability metrics
        expected_metrics = [
            'flesch_reading_ease', 'flesch_kincaid_grade',
            'average_sentence_length', 'average_word_length',
            'complex_word_percentage', 'readability_score'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, readability_result)
        
        # Readability score should be reasonable for professional content
        readability_score = readability_result.get('readability_score', 0)
        self.assertGreaterEqual(readability_score, 60)
        self.assertLessEqual(readability_score, 100)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_keyword_optimization_analysis(self, mock_config, mock_logger):
        """Test keyword optimization analysis."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test keyword analysis with job description
        keyword_analysis = qa_agent.analyze_keyword_optimization(
            cv_data=self.high_quality_cv_data,
            job_data=self.sample_job_data
        )
        
        self.assertIsInstance(keyword_analysis, dict)
        
        # Check keyword metrics
        expected_fields = [
            'keyword_density', 'relevant_keywords_found',
            'missing_keywords', 'keyword_distribution',
            'optimization_score', 'recommendations'
        ]
        
        for field in expected_fields:
            self.assertIn(field, keyword_analysis)
        
        # Check specific keyword findings
        relevant_keywords = keyword_analysis.get('relevant_keywords_found', [])
        self.assertIn('JavaScript', relevant_keywords)
        self.assertIn('Python', relevant_keywords)
        self.assertIn('React', relevant_keywords)
        
        # Optimization score should be reasonable
        optimization_score = keyword_analysis.get('optimization_score', 0)
        self.assertGreaterEqual(optimization_score, 0)
        self.assertLessEqual(optimization_score, 100)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_action_verb_analysis(self, mock_config, mock_logger):
        """Test action verb usage analysis."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test action verb analysis
        action_verb_analysis = qa_agent.analyze_action_verbs(self.high_quality_cv_data)
        
        self.assertIsInstance(action_verb_analysis, dict)
        
        # Check analysis components
        expected_fields = [
            'action_verbs_found', 'action_verb_count',
            'weak_verbs_found', 'action_verb_score',
            'recommendations', 'verb_diversity'
        ]
        
        for field in expected_fields:
            self.assertIn(field, action_verb_analysis)
        
        # Check for strong action verbs in high-quality CV
        action_verbs_found = action_verb_analysis.get('action_verbs_found', [])
        strong_verbs = ['developed', 'led', 'improved', 'implemented', 'built']
        
        found_strong_verbs = [verb for verb in strong_verbs if verb in action_verbs_found]
        self.assertGreater(len(found_strong_verbs), 2)
        
        # Action verb score should be high for well-written CV
        action_verb_score = action_verb_analysis.get('action_verb_score', 0)
        self.assertGreater(action_verb_score, 70)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_quantification_analysis(self, mock_config, mock_logger):
        """Test quantification and metrics analysis."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test quantification analysis
        quantification_analysis = qa_agent.analyze_quantification(self.high_quality_cv_data)
        
        self.assertIsInstance(quantification_analysis, dict)
        
        # Check analysis components
        expected_fields = [
            'quantified_achievements', 'quantification_count',
            'quantification_score', 'missing_quantification',
            'recommendations', 'metrics_found'
        ]
        
        for field in expected_fields:
            self.assertIn(field, quantification_analysis)
        
        # Check for quantified achievements
        quantified_achievements = quantification_analysis.get('quantified_achievements', [])
        self.assertGreater(len(quantified_achievements), 0)
        
        # Should find specific metrics in high-quality CV
        metrics_found = quantification_analysis.get('metrics_found', [])
        expected_metrics = ['40%', '100K+', '4 developers', '60%', '10K+', '95%']
        
        found_metrics = [metric for metric in expected_metrics if any(metric in achievement for achievement in quantified_achievements)]
        self.assertGreater(len(found_metrics), 2)
        
        # Quantification score should be high
        quantification_score = quantification_analysis.get('quantification_score', 0)
        self.assertGreater(quantification_score, 75)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_consistency_checking(self, mock_config, mock_logger):
        """Test consistency checking across CV sections."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test consistency analysis
        consistency_analysis = qa_agent.check_consistency(self.high_quality_cv_data)
        
        self.assertIsInstance(consistency_analysis, dict)
        
        # Check consistency components
        expected_fields = [
            'date_consistency', 'skill_consistency',
            'title_consistency', 'format_consistency',
            'consistency_score', 'inconsistencies_found'
        ]
        
        for field in expected_fields:
            self.assertIn(field, consistency_analysis)
        
        # High-quality CV should have good consistency
        consistency_score = consistency_analysis.get('consistency_score', 0)
        self.assertGreater(consistency_score, 80)
        
        # Should have minimal inconsistencies
        inconsistencies = consistency_analysis.get('inconsistencies_found', [])
        self.assertLess(len(inconsistencies), 3)

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_improvement_recommendations(self, mock_config, mock_logger):
        """Test generation of improvement recommendations."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Mock LLM response for recommendations
        mock_llm_response = LLMResponse(
            content="""{
                "priority_recommendations": [
                    {
                        "category": "Skills Enhancement",
                        "priority": "High",
                        "recommendation": "Add Kubernetes to skills section to match job requirements",
                        "impact": "Increases job alignment score by 8-12 points"
                    },
                    {
                        "category": "Experience Quantification",
                        "priority": "Medium",
                        "recommendation": "Add specific metrics to team leadership achievements",
                        "impact": "Improves quantification score and demonstrates leadership impact"
                    }
                ],
                "quick_wins": [
                    "Fix email format in contact information",
                    "Add LinkedIn profile URL",
                    "Include GPA if above 3.5"
                ],
                "long_term_improvements": [
                    "Develop more diverse project portfolio",
                    "Obtain relevant certifications",
                    "Gain experience in emerging technologies"
                ]
            }""",
            tokens_used=400,
            processing_time=2.5,
            model_used="groq",
            success=True
        )
        
        self.mock_llm_service.generate_content = AsyncMock(return_value=mock_llm_response)
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test recommendation generation
        async def test_recommendations():
            recommendations = await qa_agent.generate_improvement_recommendations(
                cv_data=self.low_quality_cv_data,
                job_data=self.sample_job_data
            )
            
            # Verify recommendations structure
            self.assertIsInstance(recommendations, dict)
            self.assertTrue(recommendations.get('success', False))
            
            rec_data = recommendations.get('data')
            self.assertIsNotNone(rec_data)
            
            # Check recommendation categories
            self.assertIn('priority_recommendations', rec_data)
            self.assertIn('quick_wins', rec_data)
            self.assertIn('long_term_improvements', rec_data)
            
            # Verify priority recommendations structure
            priority_recs = rec_data.get('priority_recommendations', [])
            self.assertGreater(len(priority_recs), 0)
            
            first_rec = priority_recs[0]
            self.assertIn('category', first_rec)
            self.assertIn('priority', first_rec)
            self.assertIn('recommendation', first_rec)
            self.assertIn('impact', first_rec)
            
            # Check quick wins
            quick_wins = rec_data.get('quick_wins', [])
            self.assertIsInstance(quick_wins, list)
            self.assertGreater(len(quick_wins), 0)
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_recommendations())
        finally:
            loop.close()

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_error_handling_in_qa_operations(self, mock_config, mock_logger):
        """Test error handling in QA operations."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Mock LLM service to raise an exception
        self.mock_llm_service.generate_content = AsyncMock(
            side_effect=Exception("LLM service error")
        )
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test error handling in job alignment analysis
        async def test_error_handling():
            result = await qa_agent.analyze_job_alignment(
                cv_data=self.high_quality_cv_data,
                job_data=self.sample_job_data
            )
            
            # Verify error handling
            self.assertIsInstance(result, dict)
            self.assertFalse(result.get('success', True))
            self.assertIsNotNone(result.get('error_message'))
            self.assertIn("error", result.get('error_message', '').lower())
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_error_handling())
        finally:
            loop.close()
        
        # Test error handling with invalid data
        invalid_cv_data = None
        
        try:
            qa_agent.validate_content_quality(invalid_cv_data)
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError, AttributeError))

    @patch('src.config.logging_config.get_structured_logger')
    @patch('src.config.settings.get_config')
    def test_quality_score_aggregation(self, mock_config, mock_logger):
        """Test quality score aggregation and weighting."""
        # Setup mocks
        mock_config.return_value = self.mock_config
        mock_logger.return_value = self.mock_logger
        
        # Create QA agent
        qa_agent = QualityAssuranceAgent(name="QualityAssuranceAgent", description="Quality assurance agent for CV content validation")
        
        # Test quality score aggregation
        individual_scores = {
            'completeness_score': 90,
            'clarity_score': 85,
            'relevance_score': 88,
            'formatting_score': 92,
            'quantification_score': 80,
            'action_verb_score': 87,
            'keyword_optimization_score': 83
        }
        
        # Test weighted aggregation
        overall_score = qa_agent.calculate_overall_quality_score(individual_scores)
        
        self.assertIsInstance(overall_score, (int, float))
        self.assertGreaterEqual(overall_score, 0)
        self.assertLessEqual(overall_score, 100)
        
        # Overall score should be reasonable average of individual scores
        expected_range_min = min(individual_scores.values()) - 5
        expected_range_max = max(individual_scores.values()) + 5
        
        self.assertGreaterEqual(overall_score, expected_range_min)
        self.assertLessEqual(overall_score, expected_range_max)
        
        # Test with different weights
        custom_weights = {
            'completeness_score': 0.25,
            'clarity_score': 0.20,
            'relevance_score': 0.20,
            'formatting_score': 0.10,
            'quantification_score': 0.15,
            'action_verb_score': 0.05,
            'keyword_optimization_score': 0.05
        }
        
        weighted_score = qa_agent.calculate_weighted_quality_score(
            individual_scores, custom_weights
        )
        
        self.assertIsInstance(weighted_score, (int, float))
        self.assertGreaterEqual(weighted_score, 0)
        self.assertLessEqual(weighted_score, 100)


if __name__ == '__main__':
    unittest.main()