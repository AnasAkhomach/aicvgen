#!/usr/bin/env python3
"""
E2E tests for complete application workflow.
Tests the full user journey from input to output generation.
"""

import pytest
import asyncio
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, CVSection, CVItem, JobDescriptionData,
    Section, Item, ItemStatus, ItemType
)
from src.services.llm import EnhancedLLMService
from tests.e2e.mock_llm_service import MockLLMService
from tests.e2e.conftest import (
    sample_job_descriptions, sample_base_cvs, 
    mock_llm_responses, expected_outputs
)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestApplicationWorkflow:
    """E2E tests for complete application workflow."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_orchestrator_llm(self):
        """Mock LLM service for orchestrator testing."""
        mock_service = MockLLMService()
        return mock_service

    @pytest.fixture
    def realistic_software_engineer_scenario(self):
        """Complete realistic scenario for software engineer."""
        job_description = """
        Senior Full-Stack Developer
        TechStartup Inc. | Remote | $120k-150k
        
        We're looking for a senior full-stack developer to join our growing team.
        
        Required Skills:
        • 5+ years of professional development experience
        • Expert-level JavaScript/TypeScript and Python
        • React, Node.js, and modern web frameworks
        • Database design and optimization (PostgreSQL, Redis)
        • Cloud platforms (AWS, Docker, Kubernetes)
        • RESTful API design and microservices architecture
        
        Responsibilities:
        • Lead development of scalable web applications
        • Mentor junior developers and conduct code reviews
        • Collaborate with product team on feature planning
        • Optimize application performance and scalability
        • Implement CI/CD pipelines and DevOps practices
        
        Company Culture:
        Fast-paced startup environment with focus on innovation,
        continuous learning, and work-life balance.
        """
        
        cv_data = StructuredCV(
            id="swe-candidate-001",
            metadata={
                "name": "Alex Johnson",
                "email": "alex.johnson@email.com",
                "phone": "+1-555-0199",
                "linkedin": "https://linkedin.com/in/alexjohnson",
                "github": "https://github.com/alexj",
                "location": "San Francisco, CA"
            },
            sections=[
                CVSection(
                    name="Technical Skills",
                    items=[
                        CVItem(id="skill_1", content="JavaScript/TypeScript"),
                        CVItem(id="skill_2", content="Python"),
                        CVItem(id="skill_3", content="React/Vue.js"),
                        CVItem(id="skill_4", content="Node.js/Express"),
                        CVItem(id="skill_5", content="PostgreSQL/MongoDB"),
                        CVItem(id="skill_6", content="AWS/Docker")
                    ]
                ),
                CVSection(
                    name="Professional Experience",
                    items=[
                        CVItem(id="exp_1", content="Senior Software Engineer @ CloudTech (2021-2024)"),
                        CVItem(id="exp_2", content="Led development of microservices platform serving 5M+ users"),
                        CVItem(id="exp_3", content="Implemented CI/CD pipelines reducing deployment time by 70%"),
                        CVItem(id="exp_4", content="Mentored team of 4 junior developers"),
                        CVItem(id="exp_5", content="Software Engineer @ WebCorp (2019-2021)"),
                        CVItem(id="exp_6", content="Built full-stack applications using React and Node.js"),
                        CVItem(id="exp_7", content="Optimized database queries improving performance by 50%")
                    ]
                ),
                CVSection(
                    name="Education",
                    items=[
                        CVItem(id="edu_1", content="BS Computer Science, Stanford University (2019)")
                    ]
                ),
                CVSection(
                    name="Projects",
                    items=[
                        CVItem(id="proj_1", content="Open-source React component library (3k+ GitHub stars)"),
                        CVItem(id="proj_2", content="Real-time collaboration platform using WebSockets")
                    ]
                )
            ]
        )
        
        return {
            "job_description": job_description,
            "cv_data": cv_data,
            "expected_skills": ["JavaScript", "TypeScript", "Python", "React", "Node.js", "AWS"],
            "expected_quality_score": 85
        }

    @pytest.fixture
    def realistic_data_scientist_scenario(self):
        """Complete realistic scenario for data scientist."""
        job_description = """
        Senior Data Scientist - ML Platform
        DataCorp | New York, NY | $140k-180k
        
        Join our ML platform team to build next-generation AI solutions.
        
        Required Qualifications:
        • PhD/MS in Computer Science, Statistics, or related field
        • 4+ years of hands-on machine learning experience
        • Expert Python programming with pandas, scikit-learn, TensorFlow
        • Strong statistical analysis and experimental design
        • Experience with big data technologies (Spark, Kafka)
        • MLOps and model deployment experience
        
        Key Responsibilities:
        • Develop and deploy ML models at scale
        • Design and execute A/B tests and experiments
        • Collaborate with engineering on model productionization
        • Analyze large datasets for business insights
        • Present findings to executive stakeholders
        
        Preferred Experience:
        • Deep learning and neural networks
        • Cloud ML platforms (AWS SageMaker, GCP AI)
        • Publications in top-tier ML conferences
        """
        
        cv_data = StructuredCV(
            id="ds-candidate-001",
            metadata={
                "name": "Dr. Maria Rodriguez",
                "email": "maria.rodriguez@email.com",
                "phone": "+1-555-0288",
                "linkedin": "https://linkedin.com/in/mariarodriguez",
                "location": "Boston, MA"
            },
            sections=[
                CVSection(
                    name="Technical Skills",
                    items=[
                        CVItem(id="skill_1", content="Python/R Programming"),
                        CVItem(id="skill_2", content="Machine Learning"),
                        CVItem(id="skill_3", content="TensorFlow/PyTorch"),
                        CVItem(id="skill_4", content="Statistical Analysis"),
                        CVItem(id="skill_5", content="SQL/NoSQL Databases"),
                        CVItem(id="skill_6", content="Data Visualization")
                    ]
                ),
                CVSection(
                    name="Professional Experience",
                    items=[
                        CVItem(id="exp_1", content="Senior Data Scientist @ MLTech (2020-2024)"),
                        CVItem(id="exp_2", content="Developed recommendation system increasing user engagement by 25%"),
                        CVItem(id="exp_3", content="Led A/B testing framework serving 10M+ users"),
                        CVItem(id="exp_4", content="Data Scientist @ AnalyticsCorp (2018-2020)"),
                        CVItem(id="exp_5", content="Built predictive models for customer churn reduction")
                    ]
                ),
                CVSection(
                    name="Education",
                    items=[
                        CVItem(id="edu_1", content="PhD Computer Science, MIT (2018)"),
                        CVItem(id="edu_2", content="MS Statistics, MIT (2015)")
                    ]
                ),
                CVSection(
                    name="Publications",
                    items=[
                        CVItem(id="pub_1", content="'Deep Learning for Time Series Forecasting' - NeurIPS 2023"),
                        CVItem(id="pub_2", content="'Scalable ML Pipelines' - ICML 2022")
                    ]
                )
            ]
        )
        
        return {
            "job_description": job_description,
            "cv_data": cv_data,
            "expected_skills": ["Python", "Machine Learning", "TensorFlow", "Statistics", "Spark"],
            "expected_quality_score": 90
        }

    async def test_complete_software_engineer_workflow(self, realistic_software_engineer_scenario, temp_output_dir, mock_orchestrator_llm):
        """Test complete workflow for software engineer scenario."""
        scenario = realistic_software_engineer_scenario
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_orchestrator_llm):
            # Create orchestrator
            orchestrator = EnhancedOrchestrator()
            
            # Create initial state
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=scenario["job_description"]),
                structured_cv=scenario["cv_data"],
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            # Execute complete workflow
            result = await orchestrator.process_cv_generation(
                initial_state,
                output_dir=temp_output_dir,
                format_type="professional"
            )
            
            # Verify workflow completed successfully
            assert "error_messages" in result
            assert len(result["error_messages"]) == 0, f"Workflow errors: {result['error_messages']}"
            
            # Verify all workflow stages completed
            self._verify_parser_completion(result, scenario["expected_skills"])
            self._verify_research_completion(result)
            self._verify_content_enhancement_completion(result)
            self._verify_quality_assessment_completion(result, scenario["expected_quality_score"])
            self._verify_output_generation_completion(result, temp_output_dir)

    async def test_complete_data_scientist_workflow(self, realistic_data_scientist_scenario, temp_output_dir, mock_orchestrator_llm):
        """Test complete workflow for data scientist scenario."""
        scenario = realistic_data_scientist_scenario
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_orchestrator_llm):
            orchestrator = EnhancedOrchestrator()
            
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=scenario["job_description"]),
                structured_cv=scenario["cv_data"],
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            result = await orchestrator.process_cv_generation(
                initial_state,
                output_dir=temp_output_dir,
                format_type="modern"
            )
            
            # Verify successful completion
            assert len(result["error_messages"]) == 0
            
            # Verify workflow stages
            self._verify_parser_completion(result, scenario["expected_skills"])
            self._verify_research_completion(result)
            self._verify_content_enhancement_completion(result)
            self._verify_quality_assessment_completion(result, scenario["expected_quality_score"])
            self._verify_output_generation_completion(result, temp_output_dir)

    async def test_workflow_with_skill_gaps(self, realistic_software_engineer_scenario, temp_output_dir, mock_orchestrator_llm):
        """Test workflow when candidate has skill gaps."""
        scenario = realistic_software_engineer_scenario
        
        # Modify CV to have fewer relevant skills
        limited_cv = scenario["cv_data"]
        limited_cv.sections[0].items = [
            CVItem(id="skill_1", content="Python"),
            CVItem(id="skill_2", content="Basic Web Development")
        ]
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_orchestrator_llm):
            orchestrator = EnhancedOrchestrator()
            
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=scenario["job_description"]),
                structured_cv=limited_cv,
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            result = await orchestrator.process_cv_generation(
                initial_state,
                output_dir=temp_output_dir,
                format_type="professional"
            )
            
            # Verify workflow completed despite skill gaps
            assert len(result["error_messages"]) == 0
            
            # Verify quality scores reflect skill gaps
            quality_scores = result.get("quality_scores", {})
            if "skill_alignment" in quality_scores:
                assert quality_scores["skill_alignment"] < 70  # Lower score expected
            
            # Verify recommendations address skill gaps
            if "qa_recommendations" in result:
                recommendations = str(result["qa_recommendations"]).lower()
                assert any(keyword in recommendations for keyword in 
                          ["skill", "experience", "develop", "learn", "improve"])

    async def test_workflow_performance_benchmarks(self, realistic_software_engineer_scenario, temp_output_dir, mock_orchestrator_llm):
        """Test workflow performance and timing benchmarks."""
        scenario = realistic_software_engineer_scenario
        
        # Add realistic delays to mock LLM
        original_generate = mock_orchestrator_llm.generate_content_async
        
        async def delayed_generate(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay per call
            return await original_generate(*args, **kwargs)
        
        mock_orchestrator_llm.generate_content_async = delayed_generate
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_orchestrator_llm):
            orchestrator = EnhancedOrchestrator()
            
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=scenario["job_description"]),
                structured_cv=scenario["cv_data"],
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            # Measure execution time
            start_time = asyncio.get_event_loop().time()
            
            result = await orchestrator.process_cv_generation(
                initial_state,
                output_dir=temp_output_dir,
                format_type="professional"
            )
            
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            # Verify reasonable execution time (should be under 10 seconds with mocked delays)
            assert execution_time < 10.0, f"Workflow took too long: {execution_time:.2f}s"
            
            # Verify successful completion
            assert len(result["error_messages"]) == 0

    async def test_workflow_error_recovery(self, realistic_software_engineer_scenario, temp_output_dir):
        """Test workflow error recovery and resilience."""
        scenario = realistic_software_engineer_scenario
        
        # Create LLM service that fails intermittently
        mock_llm = AsyncMock(spec=EnhancedLLMService)
        call_count = 0
        
        async def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Intermittent LLM failure")
            return "Successful response"
        
        mock_llm.generate_content_async.side_effect = intermittent_failure
        mock_llm.generate_structured_content_async.return_value = {
            "skills": ["Python", "JavaScript"],
            "requirements": ["5+ years experience"]
        }
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_llm):
            orchestrator = EnhancedOrchestrator()
            
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=scenario["job_description"]),
                structured_cv=scenario["cv_data"],
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            result = await orchestrator.process_cv_generation(
                initial_state,
                output_dir=temp_output_dir,
                format_type="professional"
            )
            
            # Verify workflow handled errors gracefully
            assert "error_messages" in result
            
            # Some errors may have occurred, but workflow should continue
            if len(result["error_messages"]) > 0:
                # Verify errors were logged properly
                for error in result["error_messages"]:
                    assert isinstance(error, str)
                    assert len(error) > 0
            
            # Verify some processing occurred despite errors
            assert call_count > 3  # Multiple LLM calls were attempted

    async def test_multiple_format_outputs(self, realistic_software_engineer_scenario, temp_output_dir, mock_orchestrator_llm):
        """Test generating multiple output formats."""
        scenario = realistic_software_engineer_scenario
        
        with patch('src.services.llm.EnhancedLLMService', return_value=mock_orchestrator_llm):
            orchestrator = EnhancedOrchestrator()
            
            initial_state = AgentState(
                job_description_data=JobDescriptionData(raw_text=scenario["job_description"]),
                structured_cv=scenario["cv_data"],
                error_messages=[],
                processing_queue=[],
                research_data={},
                content_data={},
                quality_scores={},
                output_data={}
            )
            
            # Test different format types
            formats_to_test = ["professional", "modern", "creative"]
            
            for format_type in formats_to_test:
                result = await orchestrator.process_cv_generation(
                    initial_state,
                    output_dir=temp_output_dir,
                    format_type=format_type
                )
                
                # Verify successful generation for each format
                assert len(result["error_messages"]) == 0, f"Errors in {format_type} format: {result['error_messages']}"
                
                # Verify output was generated
                output_data = result.get("output_data", {})
                assert len(output_data) > 0, f"No output generated for {format_type} format"

    def _verify_parser_completion(self, result, expected_skills):
        """Verify parser agent completed successfully."""
        assert "job_description_data" in result
        job_data = result["job_description_data"]
        
        # Verify job data was parsed
        assert hasattr(job_data, 'parsed_data') or 'parsed_data' in job_data.__dict__
        
        # Verify some expected skills were identified
        if hasattr(job_data, 'parsed_data') and job_data.parsed_data:
            parsed_skills = job_data.parsed_data.get("skills", [])
            if parsed_skills:
                skill_overlap = any(skill.lower() in str(parsed_skills).lower() 
                                  for skill in expected_skills[:3])  # Check first 3 skills
                assert skill_overlap, f"Expected skills {expected_skills[:3]} not found in {parsed_skills}"

    def _verify_research_completion(self, result):
        """Verify research agent completed successfully."""
        assert "research_data" in result
        research_data = result["research_data"]
        assert isinstance(research_data, dict)
        assert len(research_data) > 0
        
        # Verify research categories
        expected_categories = ["industry_trends", "skill_requirements", "company_culture"]
        found_categories = [cat for cat in expected_categories if cat in research_data]
        assert len(found_categories) > 0, f"No research categories found: {list(research_data.keys())}"

    def _verify_content_enhancement_completion(self, result):
        """Verify content enhancement completed successfully."""
        assert "content_data" in result
        content_data = result["content_data"]
        assert isinstance(content_data, dict)
        
        # Verify some content was enhanced
        if "enhanced_items" in content_data:
            assert len(content_data["enhanced_items"]) > 0

    def _verify_quality_assessment_completion(self, result, expected_min_score):
        """Verify quality assessment completed successfully."""
        assert "quality_scores" in result
        quality_scores = result["quality_scores"]
        assert isinstance(quality_scores, dict)
        
        # Verify overall score exists and is reasonable
        if "overall_score" in quality_scores:
            overall_score = quality_scores["overall_score"]
            assert isinstance(overall_score, (int, float))
            assert 0 <= overall_score <= 100
            
            # For high-quality scenarios, expect good scores
            if expected_min_score >= 85:
                assert overall_score >= 70, f"Quality score too low: {overall_score}"

    def _verify_output_generation_completion(self, result, output_dir):
        """Verify output generation completed successfully."""
        assert "output_data" in result
        output_data = result["output_data"]
        assert isinstance(output_data, dict)
        assert len(output_data) > 0
        
        # Verify output file was created or content was generated
        has_file_output = "pdf_path" in output_data or "html_path" in output_data
        has_content_output = "html_content" in output_data or "pdf_content" in output_data
        
        assert has_file_output or has_content_output, f"No output generated: {list(output_data.keys())}"
        
        # If file path is provided, verify file exists
        if "pdf_path" in output_data:
            pdf_path = output_data["pdf_path"]
            if pdf_path and os.path.exists(pdf_path):
                assert os.path.getsize(pdf_path) > 0, "Generated PDF file is empty"
        
        if "html_path" in output_data:
            html_path = output_data["html_path"]
            if html_path and os.path.exists(html_path):
                assert os.path.getsize(html_path) > 0, "Generated HTML file is empty"