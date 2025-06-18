#!/usr/bin/env python3
"""
E2E tests for realistic CV generation scenarios.
Tests complete workflows with realistic job descriptions and CV data.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, CVSection, CVItem, JobDescriptionData,
    Section, Item, ItemStatus, ItemType
)
from src.services.llm_service import EnhancedLLMService
from tests.e2e.mock_llm_service import MockLLMService


@pytest.mark.e2e
@pytest.mark.asyncio
class TestRealisticCVScenarios:
    """E2E tests for realistic CV generation scenarios."""

    @pytest.fixture
    def software_engineer_job(self):
        """Realistic software engineer job description."""
        return """
        Senior Software Engineer - Full Stack
        TechCorp Inc. | San Francisco, CA | Remote OK
        
        About the Role:
        We're seeking a Senior Software Engineer to join our growing engineering team. 
        You'll be responsible for designing and implementing scalable web applications 
        that serve millions of users worldwide.
        
        Required Skills:
        • 5+ years of professional software development experience
        • Proficiency in JavaScript/TypeScript and Python
        • Experience with React, Node.js, and modern web frameworks
        • Strong understanding of database design (PostgreSQL, MongoDB)
        • Experience with cloud platforms (AWS, GCP, or Azure)
        • Knowledge of containerization (Docker, Kubernetes)
        • Familiarity with CI/CD pipelines and DevOps practices
        
        Preferred Qualifications:
        • Experience with microservices architecture
        • Knowledge of machine learning or data science
        • Contributions to open-source projects
        • Experience leading technical teams
        
        Responsibilities:
        • Design and develop high-quality, scalable software solutions
        • Collaborate with product managers and designers on feature development
        • Mentor junior developers and conduct code reviews
        • Participate in architectural decisions and technical planning
        • Ensure code quality through testing and best practices
        
        Company Culture:
        We value innovation, collaboration, and continuous learning. Our team 
        works in an agile environment with flexible hours and strong emphasis 
        on work-life balance.
        """

    @pytest.fixture
    def data_scientist_job(self):
        """Realistic data scientist job description."""
        return """
        Senior Data Scientist - AI/ML Platform
        DataTech Solutions | New York, NY | Hybrid
        
        Position Overview:
        Join our AI/ML team to build next-generation predictive models and 
        data-driven solutions. You'll work on challenging problems across 
        recommendation systems, fraud detection, and customer analytics.
        
        Technical Requirements:
        • PhD or Master's in Computer Science, Statistics, or related field
        • 4+ years of experience in data science and machine learning
        • Expert-level Python programming with pandas, scikit-learn, TensorFlow
        • Strong statistical analysis and experimental design skills
        • Experience with big data technologies (Spark, Hadoop, Kafka)
        • Proficiency in SQL and database optimization
        • Knowledge of MLOps and model deployment practices
        
        Preferred Experience:
        • Experience with deep learning frameworks (PyTorch, TensorFlow)
        • Knowledge of cloud ML platforms (AWS SageMaker, GCP AI Platform)
        • Experience with A/B testing and causal inference
        • Publications in top-tier conferences or journals
        
        Key Responsibilities:
        • Develop and deploy machine learning models at scale
        • Analyze large datasets to extract actionable business insights
        • Design and execute experiments to validate model performance
        • Collaborate with engineering teams on model productionization
        • Present findings to stakeholders and influence product decisions
        
        What We Offer:
        Competitive salary, equity package, comprehensive benefits, and 
        opportunities to work on cutting-edge AI research with real-world impact.
        """

    @pytest.fixture
    def experienced_software_engineer_cv(self):
        """Realistic experienced software engineer CV."""
        return StructuredCV(
            id="experienced-swe-cv",
            metadata={
                "name": "Michael Chen",
                "email": "michael.chen@email.com",
                "phone": "+1-555-0123",
                "linkedin": "https://linkedin.com/in/michaelchen",
                "github": "https://github.com/mchen",
                "location": "Seattle, WA"
            },
            sections=[
                CVSection(
                    name="Key Qualifications",
                    items=[
                        CVItem(id="skill_1", content="JavaScript/TypeScript"),
                        CVItem(id="skill_2", content="Python"),
                        CVItem(id="skill_3", content="React/Node.js"),
                        CVItem(id="skill_4", content="AWS/Docker"),
                        CVItem(id="skill_5", content="PostgreSQL/MongoDB")
                    ]
                ),
                CVSection(
                    name="Professional Experience",
                    items=[
                        CVItem(id="exp_1", content="Senior Software Engineer @ CloudTech (2021-2024)"),
                        CVItem(id="exp_2", content="Led development of microservices architecture serving 10M+ users"),
                        CVItem(id="exp_3", content="Implemented CI/CD pipelines reducing deployment time by 60%"),
                        CVItem(id="exp_4", content="Software Engineer @ StartupXYZ (2019-2021)"),
                        CVItem(id="exp_5", content="Built full-stack web applications using React and Node.js"),
                        CVItem(id="exp_6", content="Optimized database queries improving application performance by 40%")
                    ]
                ),
                CVSection(
                    name="Education",
                    items=[
                        CVItem(id="edu_1", content="BS Computer Science, University of Washington (2019)")
                    ]
                ),
                CVSection(
                    name="Projects",
                    items=[
                        CVItem(id="proj_1", content="Open-source contributor to React ecosystem (2000+ GitHub stars)"),
                        CVItem(id="proj_2", content="Built real-time chat application with WebSocket and Redis")
                    ]
                )
            ]
        )

    @pytest.fixture
    def junior_data_scientist_cv(self):
        """Realistic junior data scientist CV."""
        return StructuredCV(
            id="junior-ds-cv",
            metadata={
                "name": "Sarah Rodriguez",
                "email": "sarah.rodriguez@email.com",
                "phone": "+1-555-0456",
                "linkedin": "https://linkedin.com/in/sarahrodriguez",
                "location": "Austin, TX"
            },
            sections=[
                CVSection(
                    name="Key Qualifications",
                    items=[
                        CVItem(id="skill_1", content="Python Programming"),
                        CVItem(id="skill_2", content="Machine Learning"),
                        CVItem(id="skill_3", content="Statistical Analysis"),
                        CVItem(id="skill_4", content="Data Visualization"),
                        CVItem(id="skill_5", content="SQL")
                    ]
                ),
                CVSection(
                    name="Professional Experience",
                    items=[
                        CVItem(id="exp_1", content="Data Analyst @ RetailCorp (2022-2024)"),
                        CVItem(id="exp_2", content="Analyzed customer behavior data to improve marketing campaigns"),
                        CVItem(id="exp_3", content="Built predictive models for inventory management"),
                        CVItem(id="exp_4", content="Research Assistant @ University Lab (2021-2022)"),
                        CVItem(id="exp_5", content="Conducted statistical analysis on experimental data")
                    ]
                ),
                CVSection(
                    name="Education",
                    items=[
                        CVItem(id="edu_1", content="MS Data Science, University of Texas at Austin (2022)"),
                        CVItem(id="edu_2", content="BS Mathematics, University of Texas at Austin (2020)")
                    ]
                ),
                CVSection(
                    name="Projects",
                    items=[
                        CVItem(id="proj_1", content="Kaggle competition: Top 10% in house price prediction challenge"),
                        CVItem(id="proj_2", content="Thesis: Deep learning for time series forecasting")
                    ]
                )
            ]
        )

    @pytest.fixture
    def mock_llm_responses(self):
        """Comprehensive mock LLM responses for realistic scenarios."""
        return {
            "parser_job_analysis": {
                "skills": ["JavaScript", "TypeScript", "Python", "React", "Node.js", "AWS", "Docker", "PostgreSQL"],
                "requirements": ["5+ years experience", "Full-stack development", "Cloud platforms", "Database design"],
                "responsibilities": ["Design scalable solutions", "Mentor developers", "Code reviews", "Architecture decisions"]
            },
            "research_industry_trends": """
            Current software engineering trends show strong demand for:
            - Full-stack developers with cloud expertise
            - Microservices and containerization skills
            - DevOps and CI/CD pipeline experience
            - React and modern JavaScript frameworks
            - Scalability and performance optimization
            """,
            "research_skill_requirements": """
            Key skills for senior software engineers:
            - JavaScript/TypeScript proficiency (essential)
            - React and Node.js experience (highly valued)
            - Cloud platforms (AWS preferred)
            - Database design and optimization
            - System architecture and scalability
            - Leadership and mentoring abilities
            """,
            "research_company_culture": """
            TechCorp values:
            - Innovation and technical excellence
            - Collaborative team environment
            - Continuous learning and growth
            - Work-life balance and flexibility
            - Agile development practices
            """,
            "enhanced_experience": """
            • Led development of microservices architecture serving 10M+ users, 
              implementing scalable solutions that improved system performance by 45% 
              and reduced infrastructure costs by $200K annually
            • Designed and implemented comprehensive CI/CD pipelines using Docker and 
              Kubernetes, reducing deployment time from 2 hours to 20 minutes and 
              increasing deployment frequency by 300%
            • Mentored team of 5 junior developers, conducting code reviews and 
              technical training sessions that improved team productivity by 30%
            """,
            "quality_assessment": {
                "content_relevance": 92,
                "skill_alignment": 88,
                "experience_match": 90,
                "overall_score": 90
            },
            "qa_recommendations": [
                "Highlight specific metrics and achievements in technical projects",
                "Emphasize leadership experience and team collaboration",
                "Add more details about cloud platform expertise"
            ]
        }

    def setup_mock_llm_service(self, mock_responses):
        """Set up mock LLM service with realistic responses."""
        mock_service = AsyncMock(spec=EnhancedLLMService)
        
        def get_response(prompt, **kwargs):
            prompt_lower = prompt.lower()
            
            if "industry trends" in prompt_lower:
                return mock_responses["research_industry_trends"]
            elif "skill requirements" in prompt_lower:
                return mock_responses["research_skill_requirements"]
            elif "company culture" in prompt_lower:
                return mock_responses["research_company_culture"]
            elif "enhance" in prompt_lower or "improve" in prompt_lower:
                return mock_responses["enhanced_experience"]
            elif "quality" in prompt_lower or "assess" in prompt_lower:
                return f"Score: {mock_responses['quality_assessment']['overall_score']}"
            else:
                return "Generic LLM response"
        
        def get_structured_response(prompt, response_format, **kwargs):
            prompt_lower = prompt.lower()
            
            if "parse" in prompt_lower or "extract" in prompt_lower:
                return mock_responses["parser_job_analysis"]
            elif "quality" in prompt_lower:
                return mock_responses["quality_assessment"]
            else:
                return {"result": "Generic structured response"}
        
        mock_service.generate_content_async.side_effect = get_response
        mock_service.generate_structured_content_async.side_effect = get_structured_response
        
        return mock_service

    @pytest.mark.asyncio
    async def test_software_engineer_cv_tailoring(self, software_engineer_job, experienced_software_engineer_cv, mock_llm_responses):
        """Test complete CV tailoring for software engineer position."""
        mock_llm_service = self.setup_mock_llm_service(mock_llm_responses)
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            # Create orchestrator
            orchestrator = EnhancedOrchestrator()
            
            # Create initial state
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=software_engineer_job),
                structured_cv=experienced_software_engineer_cv,
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            # Execute workflow
            with tempfile.TemporaryDirectory() as temp_dir:
                result = await orchestrator.process_cv_generation(
                    initial_state,
                    output_dir=temp_dir,
                    format_type="professional"
                )
                
                # Verify workflow completion
                assert "error_messages" in result
                assert len(result["error_messages"]) == 0, f"Errors occurred: {result['error_messages']}"
                
                # Verify research data was generated
                assert "research_data" in result
                research_data = result["research_data"]
                assert "industry_trends" in research_data
                assert "skill_requirements" in research_data
                assert "company_culture" in research_data
                
                # Verify content enhancement occurred
                assert "content_data" in result
                content_data = result["content_data"]
                assert "enhanced_items" in content_data
                assert len(content_data["enhanced_items"]) > 0
                
                # Verify quality assessment
                assert "quality_scores" in result
                quality_scores = result["quality_scores"]
                assert "overall_score" in quality_scores
                assert quality_scores["overall_score"] >= 80  # High quality expected
                
                # Verify output generation
                assert "output_data" in result
                output_data = result["output_data"]
                assert "pdf_path" in output_data or "html_content" in output_data

    @pytest.mark.asyncio
    async def test_data_scientist_cv_tailoring(self, data_scientist_job, junior_data_scientist_cv, mock_llm_responses):
        """Test CV tailoring for data scientist position with skill gap."""
        # Modify mock responses for data science scenario
        ds_responses = {
            **mock_llm_responses,
            "parser_job_analysis": {
                "skills": ["Python", "TensorFlow", "scikit-learn", "SQL", "Spark", "MLOps"],
                "requirements": ["PhD/Master's degree", "4+ years experience", "Big data technologies"],
                "responsibilities": ["Develop ML models", "Analyze datasets", "Design experiments"]
            },
            "research_industry_trends": """
            Data science trends show increasing demand for:
            - MLOps and model deployment expertise
            - Big data processing with Spark and Kafka
            - Deep learning and neural networks
            - A/B testing and experimental design
            - Cloud-based ML platforms
            """,
            "enhanced_experience": """
            • Developed and deployed machine learning models for customer behavior analysis,
              achieving 25% improvement in prediction accuracy and driving $500K in revenue
            • Built end-to-end data pipelines processing 1M+ records daily using Python and SQL,
              reducing data processing time by 60% through optimization techniques
            • Conducted A/B testing and statistical analysis to validate model performance,
              leading to data-driven decisions that improved user engagement by 15%
            """
        }
        
        mock_llm_service = self.setup_mock_llm_service(ds_responses)
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            orchestrator = EnhancedOrchestrator()
            
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=data_scientist_job),
                structured_cv=junior_data_scientist_cv,
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = await orchestrator.process_cv_generation(
                    initial_state,
                    output_dir=temp_dir,
                    format_type="modern"
                )
                
                # Verify workflow handled skill gaps appropriately
                assert len(result["error_messages"]) == 0
                
                # Verify research identified skill requirements
                research_data = result["research_data"]
                assert "MLOps" in research_data["research_industry_trends"] or \
                       "MLOps" in research_data["skill_requirements"]
                
                # Verify content enhancement addressed gaps
                content_data = result["content_data"]
                enhanced_content = str(content_data.get("enhanced_items", {}))
                assert any(keyword in enhanced_content.lower() for keyword in 
                          ["machine learning", "data", "analysis", "model"])

    @pytest.mark.asyncio
    async def test_cv_generation_with_missing_skills(self, software_engineer_job, junior_data_scientist_cv, mock_llm_responses):
        """Test CV generation when candidate lacks required skills."""
        mock_llm_service = self.setup_mock_llm_service(mock_llm_responses)
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            orchestrator = EnhancedOrchestrator()
            
            # Use data scientist CV for software engineer job (skill mismatch)
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=software_engineer_job),
                structured_cv=junior_data_scientist_cv,
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = await orchestrator.process_cv_generation(
                    initial_state,
                    output_dir=temp_dir,
                    format_type="professional"
                )
                
                # Verify workflow completed despite skill mismatch
                assert len(result["error_messages"]) == 0
                
                # Verify quality scores reflect skill gap
                quality_scores = result["quality_scores"]
                assert quality_scores.get("skill_alignment", 100) < 80  # Lower alignment expected
                
                # Verify recommendations address skill gaps
                if "qa_recommendations" in result:
                    recommendations = str(result["qa_recommendations"])
                    assert any(keyword in recommendations.lower() for keyword in 
                              ["skill", "experience", "develop", "learn"])

    @pytest.mark.asyncio
    async def test_cv_generation_performance(self, software_engineer_job, experienced_software_engineer_cv, mock_llm_responses):
        """Test CV generation performance and timing."""
        mock_llm_service = self.setup_mock_llm_service(mock_llm_responses)
        
        # Add minimal delay to simulate realistic LLM calls
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms delay
            return mock_llm_service.generate_content_async.side_effect(*args, **kwargs)
        
        mock_llm_service.generate_content_async.side_effect = delayed_response
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            orchestrator = EnhancedOrchestrator()
            
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=software_engineer_job),
                structured_cv=experienced_software_engineer_cv,
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            # Measure execution time
            start_time = asyncio.get_event_loop().time()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = await orchestrator.process_cv_generation(
                    initial_state,
                    output_dir=temp_dir,
                    format_type="professional"
                )
            
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            # Verify reasonable execution time (should be under 5 seconds for mocked LLM)
            assert execution_time < 5.0, f"Execution took too long: {execution_time:.2f}s"
            
            # Verify successful completion
            assert len(result["error_messages"]) == 0
            assert "output_data" in result

    @pytest.mark.asyncio
    async def test_cv_generation_error_recovery(self, software_engineer_job, experienced_software_engineer_cv):
        """Test CV generation error recovery mechanisms."""
        mock_llm_service = AsyncMock(spec=EnhancedLLMService)
        
        # Configure LLM to fail initially, then succeed
        call_count = 0
        
        async def failing_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 calls
                raise Exception("LLM service temporarily unavailable")
            return "Successful LLM response after retry"
        
        mock_llm_service.generate_content_async.side_effect = failing_llm_call
        mock_llm_service.generate_structured_content_async.return_value = {
            "skills": ["Python", "JavaScript"],
            "requirements": ["5+ years experience"]
        }
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm_service):
            orchestrator = EnhancedOrchestrator()
            
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=software_engineer_job),
                structured_cv=experienced_software_engineer_cv,
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = await orchestrator.process_cv_generation(
                    initial_state,
                    output_dir=temp_dir,
                    format_type="professional"
                )
                
                # Verify that some errors may have occurred but workflow continued
                # (depending on error recovery implementation)
                assert "error_messages" in result
                
                # Verify that despite initial failures, some processing occurred
                assert call_count > 2  # Retries were attempted

    def test_cv_data_validation(self, software_engineer_job, experienced_software_engineer_cv):
        """Test CV data validation and structure integrity."""
        # Verify CV structure is valid
        assert experienced_software_engineer_cv.id is not None
        assert len(experienced_software_engineer_cv.sections) > 0
        
        # Verify required sections exist
        section_names = [section.name for section in experienced_software_engineer_cv.sections]
        assert "Key Qualifications" in section_names
        assert "Professional Experience" in section_names
        
        # Verify items have proper structure
        for section in experienced_software_engineer_cv.sections:
            assert len(section.items) > 0
            for item in section.items:
                assert item.id is not None
                assert item.content is not None
                assert len(item.content.strip()) > 0

    def test_job_description_validation(self, software_engineer_job, data_scientist_job):
        """Test job description validation and parsing."""
        # Verify job descriptions contain required information
        swe_job_lower = software_engineer_job.lower()
        assert any(skill in swe_job_lower for skill in ["javascript", "python", "react"])
        assert "experience" in swe_job_lower
        assert "responsibilities" in swe_job_lower or "role" in swe_job_lower
        
        ds_job_lower = data_scientist_job.lower()
        assert any(skill in ds_job_lower for skill in ["python", "machine learning", "data"])
        assert "experience" in ds_job_lower
        assert "responsibilities" in ds_job_lower or "role" in ds_job_lower