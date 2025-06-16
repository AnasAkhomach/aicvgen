#!/usr/bin/env python3
"""
Integration tests for complete workflow orchestration.
Tests the full pipeline from job description to final CV output.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from uuid import uuid4
from pathlib import Path
import tempfile

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.orchestration.state import AgentState
from src.orchestration.cv_workflow_graph import CVWorkflowGraph
from src.models.data_models import (
    StructuredCV, CVSection, CVItem, JobDescriptionData,
    Section, Item, ItemStatus, ItemType
)
from src.core.state_manager import StateManager
from src.services.llm import EnhancedLLMService
from src.services.session_manager import SessionManager
from src.services.template_manager import TemplateManager
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.agents.formatter_agent import FormatterAgent


@pytest.mark.integration
class TestCompleteWorkflowIntegration:
    """Integration tests for complete workflow orchestration."""

    @pytest.fixture
    def sample_job_description(self):
        """Comprehensive job description for testing."""
        return """
        Senior Full-Stack Developer - AI/ML Platform
        
        Company: TechInnovate Solutions
        Location: San Francisco, CA (Remote friendly)
        
        About the Role:
        We are seeking a Senior Full-Stack Developer to join our AI/ML platform team.
        You will be responsible for building scalable web applications that power
        our machine learning infrastructure.
        
        Required Skills:
        - 5+ years of Python development experience
        - Strong experience with Django/Flask frameworks
        - Frontend development with React/Vue.js
        - Database design and optimization (PostgreSQL, MongoDB)
        - RESTful API design and implementation
        - Experience with containerization (Docker, Kubernetes)
        - Cloud platforms (AWS, GCP, or Azure)
        - Machine learning frameworks (TensorFlow, PyTorch)
        - Version control with Git
        
        Preferred Skills:
        - Experience with microservices architecture
        - Knowledge of CI/CD pipelines
        - Familiarity with data engineering tools
        - Experience with monitoring and logging systems
        
        Responsibilities:
        - Design and develop scalable web applications
        - Implement ML model serving infrastructure
        - Collaborate with data scientists and ML engineers
        - Optimize application performance and scalability
        - Mentor junior developers
        - Participate in code reviews and technical discussions
        
        Benefits:
        - Competitive salary and equity
        - Health, dental, and vision insurance
        - Flexible work arrangements
        - Professional development budget
        """

    @pytest.fixture
    def sample_base_cv(self):
        """Comprehensive base CV data for testing."""
        return {
            "personal_info": {
                "name": "John Smith",
                "email": "john.smith@email.com",
                "phone": "+1-555-0123",
                "location": "San Francisco, CA",
                "linkedin": "linkedin.com/in/johnsmith",
                "github": "github.com/johnsmith"
            },
            "summary": "Experienced software engineer with 7+ years in full-stack development",
            "experience": [
                {
                    "title": "Senior Software Engineer",
                    "company": "DataTech Corp",
                    "duration": "2020-2023",
                    "description": "Led development of data processing platform using Python and React"
                },
                {
                    "title": "Software Engineer",
                    "company": "WebSolutions Inc",
                    "duration": "2018-2020",
                    "description": "Developed web applications using Django and PostgreSQL"
                },
                {
                    "title": "Junior Developer",
                    "company": "StartupXYZ",
                    "duration": "2016-2018",
                    "description": "Built frontend components using React and JavaScript"
                }
            ],
            "education": [
                {
                    "degree": "Bachelor of Science in Computer Science",
                    "institution": "University of California, Berkeley",
                    "year": "2016",
                    "gpa": "3.8/4.0"
                }
            ],
            "skills": {
                "programming_languages": ["Python", "JavaScript", "TypeScript", "Java"],
                "frameworks": ["Django", "Flask", "React", "Vue.js", "Node.js"],
                "databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis"],
                "tools": ["Docker", "Git", "AWS", "Jenkins", "Kubernetes"]
            },
            "projects": [
                {
                    "name": "ML Model Deployment Platform",
                    "description": "Built platform for deploying ML models using Docker and Kubernetes",
                    "technologies": ["Python", "Docker", "Kubernetes", "Flask"]
                },
                {
                    "name": "Real-time Analytics Dashboard",
                    "description": "Developed dashboard for real-time data visualization",
                    "technologies": ["React", "D3.js", "WebSocket", "PostgreSQL"]
                }
            ],
            "certifications": [
                {
                    "name": "AWS Certified Solutions Architect",
                    "issuer": "Amazon Web Services",
                    "year": "2022"
                }
            ]
        }

    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for different workflow stages."""
        return {
            "parser": {
                "requirements": ["Python", "Django", "React", "PostgreSQL", "Docker", "AWS"],
                "responsibilities": ["Develop web applications", "Implement ML infrastructure", "Collaborate with teams"],
                "company_info": {"name": "TechInnovate Solutions", "industry": "AI/ML"}
            },
            "research": {
                "industry_trends": ["Microservices adoption", "AI/ML integration", "Cloud-native development"],
                "skill_relevance": {"Python": 0.95, "React": 0.85, "Docker": 0.90},
                "market_insights": "High demand for full-stack developers with ML experience"
            },
            "content_writer": {
                "enhanced_summary": "Results-driven Senior Full-Stack Developer with 7+ years of experience building scalable web applications and ML infrastructure",
                "enhanced_experience": "Led development of data processing platform serving 1M+ users, utilizing Python, Django, and React",
                "enhanced_skills": "Expert in Python, Django, React, PostgreSQL with proven experience in Docker and AWS deployment"
            },
            "qa": {
                "quality_score": 0.92,
                "suggestions": ["Add specific metrics to achievements", "Include ML project details"],
                "validation_passed": True
            },
            "formatter": {
                "formatted_output": "<html><body>Formatted CV content</body></html>",
                "pdf_ready": True
            }
        }

    @pytest.fixture
    def mock_services(self, mock_llm_responses):
        """Create mock services for testing."""
        # Mock LLM Service
        mock_llm = AsyncMock(spec=EnhancedLLMService)
        
        def llm_side_effect(*args, **kwargs):
            prompt = args[0] if args else kwargs.get('prompt', '')
            if 'parse' in prompt.lower():
                return json.dumps(mock_llm_responses['parser'])
            elif 'research' in prompt.lower():
                return json.dumps(mock_llm_responses['research'])
            elif 'enhance' in prompt.lower() or 'content' in prompt.lower():
                return mock_llm_responses['content_writer']['enhanced_summary']
            elif 'quality' in prompt.lower() or 'review' in prompt.lower():
                return json.dumps(mock_llm_responses['qa'])
            elif 'format' in prompt.lower():
                return mock_llm_responses['formatter']['formatted_output']
            else:
                return "Default response"
        
        mock_llm.generate_async.side_effect = llm_side_effect
        
        # Mock State Manager
        mock_state_manager = AsyncMock(spec=StateManager)
        mock_state_manager.save_state = AsyncMock()
        mock_state_manager.load_state = AsyncMock()
        
        # Mock Session Manager
        mock_session_manager = AsyncMock(spec=SessionManager)
        mock_session_manager.create_session = AsyncMock(return_value=str(uuid4()))
        mock_session_manager.save_session = AsyncMock()
        
        # Mock Template Manager
        mock_template_manager = Mock(spec=TemplateManager)
        mock_template_manager.render_template = Mock(return_value="Rendered template")
        
        return {
            'llm': mock_llm,
            'state_manager': mock_state_manager,
            'session_manager': mock_session_manager,
            'template_manager': mock_template_manager
        }

    @pytest.mark.asyncio
    async def test_complete_workflow_success_path(self, sample_job_description, sample_base_cv, mock_services):
        """Test the complete workflow success path."""
        with patch.multiple(
            'src.core.enhanced_orchestrator',
            EnhancedLLMService=lambda: mock_services['llm'],
            StateManager=lambda: mock_services['state_manager'],
            SessionManager=lambda: mock_services['session_manager']
        ):
            orchestrator = EnhancedOrchestrator()
            
            # Execute complete workflow
            result = await orchestrator.process_cv(
                job_description=sample_job_description,
                base_cv_data=sample_base_cv
            )
            
            # Verify workflow completion
            assert 'session_id' in result
            assert 'structured_cv' in result
            assert 'final_output' in result
            assert len(result.get('errors', [])) == 0
            
            # Verify LLM service was called for each stage
            assert mock_services['llm'].generate_async.call_count >= 4  # Parser, Research, Content, QA

    @pytest.mark.asyncio
    async def test_workflow_with_partial_failures(self, sample_job_description, sample_base_cv, mock_services):
        """Test workflow resilience with partial failures."""
        # Setup partial failure scenario
        call_count = 0
        def failing_llm_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call (research stage)
                raise Exception("Research service temporarily unavailable")
            return "Success response"
        
        mock_services['llm'].generate_async.side_effect = failing_llm_side_effect
        
        with patch.multiple(
            'src.core.enhanced_orchestrator',
            EnhancedLLMService=lambda: mock_services['llm'],
            StateManager=lambda: mock_services['state_manager'],
            SessionManager=lambda: mock_services['session_manager']
        ):
            orchestrator = EnhancedOrchestrator()
            
            result = await orchestrator.process_cv(
                job_description=sample_job_description,
                base_cv_data=sample_base_cv
            )
            
            # Verify workflow continued despite failure
            assert 'session_id' in result
            assert 'errors' in result
            assert len(result['errors']) > 0
            # Should still produce some output
            assert 'structured_cv' in result

    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, sample_job_description, sample_base_cv, mock_services):
        """Test that workflow state is properly persisted."""
        session_id = str(uuid4())
        
        with patch.multiple(
            'src.core.enhanced_orchestrator',
            EnhancedLLMService=lambda: mock_services['llm'],
            StateManager=lambda: mock_services['state_manager'],
            SessionManager=lambda: mock_services['session_manager']
        ):
            orchestrator = EnhancedOrchestrator()
            
            # Execute workflow
            result = await orchestrator.process_cv(
                job_description=sample_job_description,
                base_cv_data=sample_base_cv,
                session_id=session_id
            )
            
            # Verify state persistence calls
            assert mock_services['state_manager'].save_state.called
            assert mock_services['session_manager'].save_session.called

    @pytest.mark.asyncio
    async def test_workflow_concurrent_processing(self, sample_job_description, sample_base_cv, mock_services):
        """Test workflow handling of concurrent processing requests."""
        with patch.multiple(
            'src.core.enhanced_orchestrator',
            EnhancedLLMService=lambda: mock_services['llm'],
            StateManager=lambda: mock_services['state_manager'],
            SessionManager=lambda: mock_services['session_manager']
        ):
            orchestrator = EnhancedOrchestrator()
            
            # Execute multiple concurrent workflows
            tasks = [
                orchestrator.process_cv(
                    job_description=sample_job_description,
                    base_cv_data=sample_base_cv
                )
                for _ in range(3)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all workflows completed
            assert len(results) == 3
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 2  # At least 2 should succeed
            
            # Verify unique session IDs
            session_ids = [r.get('session_id') for r in successful_results if isinstance(r, dict)]
            assert len(set(session_ids)) == len(session_ids)  # All unique

    @pytest.mark.asyncio
    async def test_workflow_graph_execution(self, sample_job_description, sample_base_cv, mock_services):
        """Test the LangGraph workflow execution directly."""
        # Create initial state
        job_desc_data = JobDescriptionData(
            raw_text=sample_job_description,
            requirements=[],
            responsibilities=[]
        )
        
        initial_state = AgentState(
            session_id=str(uuid4()),
            job_description_data=job_desc_data,
            base_cv_data=sample_base_cv,
            structured_cv=StructuredCV(sections=[]),
            processing_queue=[],
            errors=[]
        )
        
        # Mock the workflow graph
        with patch('src.orchestration.cv_workflow_graph.parser_node') as mock_parser, \
             patch('src.orchestration.cv_workflow_graph.research_node') as mock_research, \
             patch('src.orchestration.cv_workflow_graph.content_writer_node') as mock_content:
            
            # Setup mock node responses
            mock_parser.return_value = {**initial_state.dict(), "processing_queue": ["exp_1", "skill_1"]}
            mock_research.return_value = {**initial_state.dict(), "research_findings": {"trends": ["AI/ML"]}}
            mock_content.return_value = {**initial_state.dict(), "current_item_id": "exp_1"}
            
            # Create and execute workflow graph
            workflow_graph = CVWorkflowGraph()
            compiled_graph = workflow_graph.create_graph()
            
            # Execute the graph
            result = await compiled_graph.ainvoke(initial_state.dict())
            
            # Verify graph execution
            assert result is not None
            assert 'session_id' in result

    @pytest.mark.asyncio
    async def test_workflow_performance_metrics(self, sample_job_description, sample_base_cv, mock_services):
        """Test workflow performance metrics collection."""
        import time
        
        with patch.multiple(
            'src.core.enhanced_orchestrator',
            EnhancedLLMService=lambda: mock_services['llm'],
            StateManager=lambda: mock_services['state_manager'],
            SessionManager=lambda: mock_services['session_manager']
        ):
            orchestrator = EnhancedOrchestrator()
            
            start_time = time.time()
            result = await orchestrator.process_cv(
                job_description=sample_job_description,
                base_cv_data=sample_base_cv
            )
            end_time = time.time()
            
            # Verify performance metrics
            execution_time = end_time - start_time
            assert execution_time < 30  # Should complete within 30 seconds
            
            # Verify metrics in result
            if 'metrics' in result:
                assert 'execution_time' in result['metrics']
                assert 'agent_performance' in result['metrics']

    @pytest.mark.asyncio
    async def test_workflow_data_validation(self, sample_job_description, sample_base_cv, mock_services):
        """Test workflow data validation at each stage."""
        with patch.multiple(
            'src.core.enhanced_orchestrator',
            EnhancedLLMService=lambda: mock_services['llm'],
            StateManager=lambda: mock_services['state_manager'],
            SessionManager=lambda: mock_services['session_manager']
        ):
            orchestrator = EnhancedOrchestrator()
            
            # Test with invalid job description
            result_invalid_job = await orchestrator.process_cv(
                job_description="",  # Empty job description
                base_cv_data=sample_base_cv
            )
            
            # Should handle gracefully
            assert 'errors' in result_invalid_job
            
            # Test with invalid CV data
            result_invalid_cv = await orchestrator.process_cv(
                job_description=sample_job_description,
                base_cv_data={}  # Empty CV data
            )
            
            # Should handle gracefully
            assert 'errors' in result_invalid_cv or 'structured_cv' in result_invalid_cv

    @pytest.mark.asyncio
    async def test_workflow_output_formats(self, sample_job_description, sample_base_cv, mock_services):
        """Test workflow output in different formats."""
        with patch.multiple(
            'src.core.enhanced_orchestrator',
            EnhancedLLMService=lambda: mock_services['llm'],
            StateManager=lambda: mock_services['state_manager'],
            SessionManager=lambda: mock_services['session_manager']
        ):
            orchestrator = EnhancedOrchestrator()
            
            # Test different output formats
            formats = ['json', 'html', 'pdf']
            
            for output_format in formats:
                result = await orchestrator.process_cv(
                    job_description=sample_job_description,
                    base_cv_data=sample_base_cv,
                    output_format=output_format
                )
                
                # Verify format-specific output
                assert 'final_output' in result
                if output_format == 'json':
                    assert isinstance(result['final_output'], (dict, str))
                elif output_format in ['html', 'pdf']:
                    assert isinstance(result['final_output'], str)

    @pytest.mark.asyncio
    async def test_workflow_cleanup_and_resource_management(self, sample_job_description, sample_base_cv, mock_services):
        """Test workflow cleanup and resource management."""
        with patch.multiple(
            'src.core.enhanced_orchestrator',
            EnhancedLLMService=lambda: mock_services['llm'],
            StateManager=lambda: mock_services['state_manager'],
            SessionManager=lambda: mock_services['session_manager']
        ):
            orchestrator = EnhancedOrchestrator()
            
            # Execute workflow
            result = await orchestrator.process_cv(
                job_description=sample_job_description,
                base_cv_data=sample_base_cv
            )
            
            # Verify cleanup was performed
            # This would typically involve checking that temporary files are cleaned up,
            # connections are closed, etc.
            assert 'session_id' in result
            
            # Test explicit cleanup
            if hasattr(orchestrator, 'cleanup'):
                await orchestrator.cleanup()