"""End-to-End Test 1: Complete CV Generation

Tests the complete workflow from job description + base CV (Markdown)
through all agents to produce a tailored PDF with "Big 10" skills,
experience bullets, and projects.
"""

import pytest
import asyncio
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Add project root to path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.models.data_models import (
    CVGenerationState, WorkflowStage, ProcessingStatus, ContentType,
    JobDescriptionData, ExperienceItem, ProjectItem, QualificationItem
)
from src.services.session_manager import SessionManager
from src.services.progress_tracker import ProgressTracker
from src.services.error_recovery import ErrorRecoveryService


@pytest.mark.e2e
@pytest.mark.slow
class TestCompleteCVGeneration:
    """End-to-end tests for complete CV generation workflow."""
    
    @pytest.fixture
    def sample_job_description(self) -> str:
        """Provide a realistic tech job description for testing."""
        return """
        AI Engineer - Machine Learning Platform
        TechCorp Inc.
        
        We are seeking an experienced AI Engineer to join our Machine Learning Platform team.
        You will be responsible for designing, developing, and deploying scalable ML solutions
        that serve millions of users worldwide.
        
        Key Responsibilities:
        - Design and implement machine learning models for recommendation systems
        - Develop and maintain ML pipelines using Python, TensorFlow, and PyTorch
        - Collaborate with data scientists to productionize research models
        - Optimize model performance and scalability in cloud environments
        - Implement MLOps practices for continuous model deployment
        
        Required Qualifications:
        - Master's degree in Computer Science, AI, or related field
        - 3+ years of experience in machine learning engineering
        - Strong proficiency in Python, TensorFlow/PyTorch, and SQL
        - Experience with cloud platforms (AWS, GCP, or Azure)
        - Knowledge of containerization (Docker, Kubernetes)
        - Experience with ML frameworks and model deployment
        
        Preferred Skills:
        - Experience with recommendation systems
        - Knowledge of distributed computing (Spark, Dask)
        - Familiarity with MLOps tools (MLflow, Kubeflow)
        - Strong communication and collaboration skills
        """
    
    @pytest.fixture
    def sample_base_cv(self) -> str:
        """Provide a realistic base CV in Markdown format."""
        return """
        # John Smith
        ## Contact Information
        - Email: john.smith@email.com
        - Phone: (555) 123-4567
        - LinkedIn: linkedin.com/in/johnsmith
        - GitHub: github.com/johnsmith
        
        ## Professional Experience
        
        ### Senior Software Engineer | DataTech Solutions | 2021-2023
        - Developed scalable data processing pipelines using Python and Apache Spark
        - Implemented machine learning models for customer segmentation
        - Led a team of 3 engineers in building real-time analytics platform
        - Reduced data processing time by 40% through optimization techniques
        
        ### Machine Learning Engineer | StartupAI | 2019-2021
        - Built and deployed deep learning models for computer vision applications
        - Designed MLOps infrastructure using Docker and Kubernetes
        - Collaborated with product team to integrate ML features into mobile app
        - Achieved 95% accuracy in image classification models
        
        ### Software Developer | TechStart Inc. | 2017-2019
        - Developed web applications using Python Flask and React
        - Implemented RESTful APIs for mobile and web clients
        - Worked with PostgreSQL and Redis for data storage
        - Participated in agile development processes
        
        ## Education
        
        ### Master of Science in Computer Science | University of Technology | 2017
        - Specialization: Machine Learning and Artificial Intelligence
        - Thesis: "Deep Learning Approaches for Natural Language Processing"
        - GPA: 3.8/4.0
        
        ### Bachelor of Science in Software Engineering | State University | 2015
        - Relevant Coursework: Data Structures, Algorithms, Database Systems
        - GPA: 3.6/4.0
        
        ## Technical Skills
        - Programming Languages: Python, Java, JavaScript, SQL
        - ML/AI Frameworks: TensorFlow, PyTorch, Scikit-learn, Keras
        - Cloud Platforms: AWS (EC2, S3, Lambda), Google Cloud Platform
        - Tools & Technologies: Docker, Kubernetes, Git, Jenkins
        - Databases: PostgreSQL, MongoDB, Redis
        
        ## Projects
        
        ### Recommendation Engine for E-commerce Platform
        - Built collaborative filtering system serving 100K+ users
        - Implemented using Python, TensorFlow, and AWS infrastructure
        - Increased user engagement by 25% and sales conversion by 15%
        
        ### Real-time Fraud Detection System
        - Developed ML pipeline for detecting fraudulent transactions
        - Used ensemble methods with 99.2% accuracy and <100ms latency
        - Deployed on Kubernetes cluster with auto-scaling capabilities
        
        ### Open Source NLP Library
        - Created Python library for text preprocessing and analysis
        - 500+ GitHub stars and active community contributions
        - Integrated with popular ML frameworks and cloud services
        """
    
    @pytest.fixture
    def mock_llm_responses(self) -> Dict[str, Any]:
        """Provide realistic mock LLM responses for different processing stages."""
        return {
            "big_10_skills": {
                "content": json.dumps({
                    "technical_skills": [
                        "Machine Learning Engineering",
                        "Python Programming",
                        "TensorFlow/PyTorch",
                        "Cloud Computing (AWS/GCP)",
                        "MLOps and Model Deployment"
                    ],
                    "soft_skills": [
                        "Technical Leadership",
                        "Cross-functional Collaboration",
                        "Problem Solving",
                        "Communication",
                        "Continuous Learning"
                    ]
                }),
                "tokens_used": 150,
                "model": "gpt-4"
            },
            "experience_bullets": {
                "content": json.dumps({
                    "tailored_bullets": [
                        "Architected and deployed ML recommendation systems serving 1M+ users with 99.9% uptime using TensorFlow and AWS infrastructure",
                        "Led cross-functional team of 5 engineers to build real-time ML pipeline, reducing model inference latency by 60%",
                        "Implemented MLOps practices with Docker/Kubernetes, enabling continuous model deployment and A/B testing"
                    ]
                }),
                "tokens_used": 120,
                "model": "gpt-4"
            },
            "project_enhancement": {
                "content": json.dumps({
                    "enhanced_projects": [
                        {
                            "title": "AI-Powered Recommendation Engine",
                            "description": "Designed and implemented scalable recommendation system using collaborative filtering and deep learning, serving 100K+ users with 25% engagement increase",
                            "technologies": ["Python", "TensorFlow", "AWS", "Kubernetes"]
                        }
                    ]
                }),
                "tokens_used": 100,
                "model": "gpt-4"
            }
        }
    
    @pytest.fixture
    def orchestrator_with_mocks(self, mock_llm_responses, temp_dir):
        """Provide an orchestrator with mocked dependencies."""
        # Mock LLM client
        mock_llm_client = Mock()
        mock_llm_client.generate_content = AsyncMock()
        
        # Configure mock responses based on prompt content
        def mock_generate_content(*args, **kwargs):
            prompt = kwargs.get('prompt', '') or (args[0] if args else '')
            
            if 'big 10' in prompt.lower() or 'skills' in prompt.lower():
                return mock_llm_responses['big_10_skills']
            elif 'experience' in prompt.lower() or 'bullet' in prompt.lower():
                return mock_llm_responses['experience_bullets']
            elif 'project' in prompt.lower():
                return mock_llm_responses['project_enhancement']
            else:
                return {
                    "content": "Generated content",
                    "tokens_used": 50,
                    "model": "gpt-4"
                }
        
        mock_llm_client.generate_content.side_effect = mock_generate_content
        
        # Mock services
        mock_progress_tracker = Mock(spec=ProgressTracker)
        mock_error_recovery = Mock(spec=ErrorRecoveryService)
        mock_session_manager = Mock(spec=SessionManager)
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator(
            llm_client=mock_llm_client,
            progress_tracker=mock_progress_tracker,
            error_recovery=mock_error_recovery,
            session_manager=mock_session_manager
        )
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_complete_cv_generation_workflow(
        self,
        orchestrator_with_mocks,
        sample_job_description,
        sample_base_cv,
        performance_timer
    ):
        """Test E2E Test 1: Complete CV Generation workflow.
        
        Input: Sample job description + base CV (Markdown)
        Process: Full workflow through all agents
        Output: Tailored PDF with "Big 10" skills, experience bullets, projects
        Assertions: PDF structure, content quality, processing time
        """
        orchestrator = orchestrator_with_mocks
        
        # Start performance timer
        performance_timer.start()
        
        # Create job description data
        job_data = JobDescriptionData(
            title="AI Engineer - Machine Learning Platform",
            company="TechCorp Inc.",
            description=sample_job_description,
            requirements=[
                "Master's degree in Computer Science, AI, or related field",
                "3+ years of experience in machine learning engineering",
                "Strong proficiency in Python, TensorFlow/PyTorch, and SQL"
            ],
            responsibilities=[
                "Design and implement machine learning models",
                "Develop and maintain ML pipelines",
                "Optimize model performance and scalability"
            ],
            skills=[
                "Python", "TensorFlow", "PyTorch", "AWS", "MLOps"
            ]
        )
        
        # Process complete CV generation
        try:
            result = await orchestrator.process_complete_cv(
                job_description=job_data,
                base_cv_content=sample_base_cv,
                session_id="test-e2e-complete"
            )
            
            # Stop performance timer
            performance_timer.stop()
            
            # Assertions: Processing completed successfully
            assert result is not None
            assert result.get('status') == 'completed'
            
            # Assertions: Content quality
            generated_content = result.get('generated_content', {})
            
            # Check for "Big 10" skills
            skills_section = generated_content.get('skills', {})
            assert 'technical_skills' in skills_section
            assert 'soft_skills' in skills_section
            assert len(skills_section['technical_skills']) >= 5
            assert len(skills_section['soft_skills']) >= 5
            
            # Check for enhanced experience bullets
            experience_section = generated_content.get('experience', [])
            assert len(experience_section) > 0
            
            # Verify experience bullets are tailored (should contain job-relevant keywords)
            experience_text = str(experience_section).lower()
            assert any(keyword in experience_text for keyword in [
                'machine learning', 'tensorflow', 'aws', 'mlops', 'python'
            ])
            
            # Check for enhanced projects
            projects_section = generated_content.get('projects', [])
            assert len(projects_section) > 0
            
            # Assertions: Processing time (should complete within 2 minutes)
            assert performance_timer.elapsed < 120.0, f"Processing took {performance_timer.elapsed:.2f}s, expected < 120s"
            
            # Assertions: PDF structure (mock validation)
            pdf_metadata = result.get('pdf_metadata', {})
            assert pdf_metadata.get('format') == 'PDF'
            assert pdf_metadata.get('pages') > 0
            
            print(f"✅ Complete CV generation test passed in {performance_timer.elapsed:.2f}s")
            
        except Exception as e:
            performance_timer.stop()
            pytest.fail(f"Complete CV generation failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_content_quality_validation(
        self,
        orchestrator_with_mocks,
        sample_job_description,
        sample_base_cv
    ):
        """Test content quality and relevance of generated CV sections."""
        orchestrator = orchestrator_with_mocks
        
        job_data = JobDescriptionData(
            title="AI Engineer",
            company="TechCorp",
            description=sample_job_description,
            requirements=["Python", "Machine Learning", "AWS"],
            responsibilities=["Build ML models", "Deploy systems"],
            skills=["TensorFlow", "PyTorch", "MLOps"]
        )
        
        result = await orchestrator.process_complete_cv(
            job_description=job_data,
            base_cv_content=sample_base_cv,
            session_id="test-e2e-quality"
        )
        
        # Quality assertions
        generated_content = result.get('generated_content', {})
        
        # Skills should be relevant to job requirements
        skills = generated_content.get('skills', {})
        technical_skills = [skill.lower() for skill in skills.get('technical_skills', [])]
        
        # Should contain job-relevant technical skills
        job_relevant_skills = ['python', 'machine learning', 'tensorflow', 'aws']
        relevant_count = sum(1 for skill in job_relevant_skills 
                           if any(skill in tech_skill for tech_skill in technical_skills))
        
        assert relevant_count >= 2, f"Expected at least 2 job-relevant skills, found {relevant_count}"
        
        # Experience bullets should be quantified and impactful
        experience = generated_content.get('experience', [])
        experience_text = ' '.join(str(exp) for exp in experience)
        
        # Should contain quantified achievements
        quantifiers = ['%', 'million', 'thousand', 'users', 'latency', 'accuracy']
        has_quantifiers = any(q in experience_text.lower() for q in quantifiers)
        assert has_quantifiers, "Experience bullets should contain quantified achievements"
        
        print("✅ Content quality validation passed")
    
    @pytest.mark.asyncio
    async def test_pdf_generation_structure(
        self,
        orchestrator_with_mocks,
        sample_job_description,
        sample_base_cv
    ):
        """Test PDF generation and structure validation."""
        orchestrator = orchestrator_with_mocks
        
        job_data = JobDescriptionData(
            title="AI Engineer",
            company="TechCorp",
            description=sample_job_description,
            requirements=["Python"],
            responsibilities=["Build ML models"],
            skills=["TensorFlow"]
        )
        
        # Mock PDF generation
        with patch('src.services.pdf_generator.PDFGenerator') as mock_pdf_gen:
            mock_pdf_instance = Mock()
            mock_pdf_instance.generate_pdf.return_value = {
                'success': True,
                'file_path': '/tmp/test_cv.pdf',
                'metadata': {
                    'format': 'PDF',
                    'pages': 2,
                    'size_kb': 150
                }
            }
            mock_pdf_gen.return_value = mock_pdf_instance
            
            result = await orchestrator.process_complete_cv(
                job_description=job_data,
                base_cv_content=sample_base_cv,
                session_id="test-e2e-pdf",
                generate_pdf=True
            )
            
            # PDF structure assertions
            pdf_result = result.get('pdf_result', {})
            assert pdf_result.get('success') is True
            assert pdf_result.get('file_path') is not None
            
            pdf_metadata = pdf_result.get('metadata', {})
            assert pdf_metadata.get('format') == 'PDF'
            assert pdf_metadata.get('pages') > 0
            assert pdf_metadata.get('size_kb') > 0
            
            print("✅ PDF generation structure test passed")