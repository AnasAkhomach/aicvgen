"""End-to-End Test 2: Individual Item Processing

Tests the role-by-role generation workflow (MVP requirement) with
rate limit compliance and user feedback integration.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Add project root to path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.models.data_models import (
    CVGenerationState, WorkflowStage, ProcessingStatus, ContentType,
    JobDescriptionData, ExperienceItem, ProjectItem, ContentItem,
    ProcessingMetadata
)
from src.services.rate_limiter import RateLimitConfig
from src.services.session_manager import SessionManager
from src.services.progress_tracker import ProgressTracker


@pytest.mark.e2e
@pytest.mark.slow
class TestIndividualItemProcessing:
    """End-to-end tests for individual item processing workflow."""
    
    @pytest.fixture
    def sample_experience_items(self) -> List[ExperienceItem]:
        """Provide sample professional experience items for testing."""
        return [
            ExperienceItem(
                title="Senior Software Engineer",
                company="DataTech Solutions",
                duration="2021-2023",
                responsibilities=[
                    "Developed scalable data processing pipelines using Python and Apache Spark",
                    "Implemented machine learning models for customer segmentation",
                    "Led a team of 3 engineers in building real-time analytics platform",
                    "Reduced data processing time by 40% through optimization techniques"
                ],
                technologies=["Python", "Apache Spark", "Machine Learning", "Analytics"]
            ),
            ExperienceItem(
                title="Machine Learning Engineer",
                company="StartupAI",
                duration="2019-2021",
                responsibilities=[
                    "Built and deployed deep learning models for computer vision applications",
                    "Designed MLOps infrastructure using Docker and Kubernetes",
                    "Collaborated with product team to integrate ML features into mobile app",
                    "Achieved 95% accuracy in image classification models"
                ],
                technologies=["Deep Learning", "Computer Vision", "Docker", "Kubernetes", "MLOps"]
            ),
            ExperienceItem(
                title="Software Developer",
                company="TechStart Inc.",
                duration="2017-2019",
                responsibilities=[
                    "Developed web applications using Python Flask and React",
                    "Implemented RESTful APIs for mobile and web clients",
                    "Worked with PostgreSQL and Redis for data storage",
                    "Participated in agile development processes"
                ],
                technologies=["Python", "Flask", "React", "PostgreSQL", "Redis"]
            )
        ]
    
    @pytest.fixture
    def ai_engineer_job_description(self) -> JobDescriptionData:
        """Provide AI Engineer job description for testing."""
        return JobDescriptionData(
            title="AI Engineer - Machine Learning Platform",
            company="TechCorp Inc.",
            description="We are seeking an experienced AI Engineer to join our ML Platform team.",
            requirements=[
                "3+ years of experience in machine learning engineering",
                "Strong proficiency in Python, TensorFlow/PyTorch",
                "Experience with cloud platforms (AWS, GCP, or Azure)",
                "Knowledge of containerization (Docker, Kubernetes)"
            ],
            responsibilities=[
                "Design and implement machine learning models for recommendation systems",
                "Develop and maintain ML pipelines using Python, TensorFlow, and PyTorch",
                "Optimize model performance and scalability in cloud environments",
                "Implement MLOps practices for continuous model deployment"
            ],
            skills=[
                "Python", "TensorFlow", "PyTorch", "AWS", "Docker", "Kubernetes",
                "Machine Learning", "MLOps", "Data Engineering"
            ]
        )
    
    @pytest.fixture
    def mock_rate_limited_llm_client(self):
        """Provide a mock LLM client that simulates rate limiting."""
        mock_client = Mock()
        mock_client.generate_content = AsyncMock()
        
        # Track call count for rate limiting simulation
        call_count = {'count': 0}
        
        async def rate_limited_generate(*args, **kwargs):
            call_count['count'] += 1
            
            # Simulate rate limit after 5 calls
            if call_count['count'] > 5:
                await asyncio.sleep(0.1)  # Simulate rate limit delay
            
            # Return tailored content based on input
            prompt = kwargs.get('prompt', '') or (args[0] if args else '')
            
            if 'senior software engineer' in prompt.lower():
                return {
                    "content": json.dumps({
                        "tailored_bullets": [
                            "Architected scalable ML data pipelines processing 10TB+ daily using Python and Apache Spark, enabling real-time model training",
                            "Led cross-functional team of 5 engineers to build production ML platform, reducing model deployment time by 70%",
                            "Implemented advanced customer segmentation models achieving 25% improvement in targeting accuracy"
                        ]
                    }),
                    "tokens_used": 120,
                    "model": "gpt-4"
                }
            elif 'machine learning engineer' in prompt.lower():
                return {
                    "content": json.dumps({
                        "tailored_bullets": [
                            "Developed and deployed computer vision models achieving 95% accuracy for production mobile app with 1M+ users",
                            "Built comprehensive MLOps infrastructure using Docker/Kubernetes, enabling continuous model deployment and A/B testing",
                            "Collaborated with product teams to integrate ML features, resulting in 30% increase in user engagement"
                        ]
                    }),
                    "tokens_used": 115,
                    "model": "gpt-4"
                }
            else:
                return {
                    "content": json.dumps({
                        "tailored_bullets": [
                            "Enhanced web application performance and scalability using modern Python frameworks",
                            "Implemented robust RESTful APIs serving mobile and web clients with 99.9% uptime",
                            "Optimized database operations and caching strategies for improved response times"
                        ]
                    }),
                    "tokens_used": 100,
                    "model": "gpt-4"
                }
        
        mock_client.generate_content.side_effect = rate_limited_generate
        return mock_client
    
    @pytest.fixture
    def orchestrator_with_rate_limiting(self, mock_rate_limited_llm_client, temp_dir):
        """Provide an orchestrator configured for rate limiting tests."""
        # Mock services
        mock_progress_tracker = Mock(spec=ProgressTracker)
        mock_session_manager = Mock(spec=SessionManager)
        
        # Create orchestrator with rate limiting
        orchestrator = EnhancedOrchestrator(
            llm_client=mock_rate_limited_llm_client,
            progress_tracker=mock_progress_tracker,
            session_manager=mock_session_manager
        )
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_individual_experience_processing(
        self,
        orchestrator_with_rate_limiting,
        sample_experience_items,
        ai_engineer_job_description,
        performance_timer
    ):
        """Test E2E Test 2: Individual Item Processing workflow.
        
        Input: Single professional experience role
        Process: Role-by-role generation (MVP requirement)
        Output: Tailored experience bullets
        Assertions: Rate limit compliance, user feedback integration
        """
        orchestrator = orchestrator_with_rate_limiting
        performance_timer.start()
        
        # Process each experience item individually
        processed_items = []
        processing_times = []
        
        for i, experience_item in enumerate(sample_experience_items):
            item_start_time = performance_timer.elapsed
            
            try:
                # Process individual experience item
                result = await orchestrator.process_individual_item(
                    item=experience_item,
                    job_description=ai_engineer_job_description,
                    session_id=f"test-individual-{i}",
                    item_type=ContentType.EXPERIENCE
                )
                
                item_end_time = performance_timer.elapsed
                processing_times.append(item_end_time - item_start_time)
                
                # Assertions: Individual item processing
                assert result is not None
                assert result.get('status') == ProcessingStatus.COMPLETED
                
                # Check tailored content
                tailored_content = result.get('generated_content', {})
                assert 'tailored_bullets' in tailored_content
                
                bullets = tailored_content['tailored_bullets']
                assert len(bullets) >= 3, f"Expected at least 3 bullets, got {len(bullets)}"
                
                # Verify content is tailored to job requirements
                bullets_text = ' '.join(bullets).lower()
                job_keywords = ['machine learning', 'python', 'ml', 'data', 'model']
                keyword_matches = sum(1 for keyword in job_keywords if keyword in bullets_text)
                assert keyword_matches >= 2, f"Expected job-relevant keywords, found {keyword_matches} matches"
                
                processed_items.append(result)
                
                print(f"✅ Processed {experience_item.title} in {processing_times[-1]:.2f}s")
                
            except Exception as e:
                pytest.fail(f"Failed to process {experience_item.title}: {str(e)}")
        
        performance_timer.stop()
        
        # Assertions: Rate limit compliance
        total_processing_time = performance_timer.elapsed
        average_processing_time = sum(processing_times) / len(processing_times)
        
        # Should respect rate limits (processing time should increase with more items)
        assert len(processed_items) == len(sample_experience_items)
        assert total_processing_time < 60.0, f"Total processing took {total_processing_time:.2f}s, expected < 60s"
        
        # Verify rate limiting behavior (later items should take slightly longer)
        if len(processing_times) > 1:
            # Allow for some variance but expect general trend
            later_items_avg = sum(processing_times[1:]) / len(processing_times[1:])
            first_item_time = processing_times[0]
            
            # Later items might be slightly slower due to rate limiting
            assert later_items_avg <= first_item_time * 2, "Rate limiting causing excessive delays"
        
        print(f"✅ Individual item processing test completed in {total_processing_time:.2f}s")
        print(f"   Average processing time per item: {average_processing_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_user_feedback_integration(
        self,
        orchestrator_with_rate_limiting,
        sample_experience_items,
        ai_engineer_job_description
    ):
        """Test user feedback integration in individual item processing."""
        orchestrator = orchestrator_with_rate_limiting
        
        # Process first experience item
        experience_item = sample_experience_items[0]
        
        # Initial processing
        initial_result = await orchestrator.process_individual_item(
            item=experience_item,
            job_description=ai_engineer_job_description,
            session_id="test-feedback-integration",
            item_type=ContentType.EXPERIENCE
        )
        
        assert initial_result.get('status') == ProcessingStatus.COMPLETED
        initial_bullets = initial_result.get('generated_content', {}).get('tailored_bullets', [])
        
        # Simulate user feedback
        user_feedback = {
            "feedback_type": "enhancement_request",
            "feedback": "Please emphasize leadership and team management aspects more",
            "item_id": initial_result.get('item_id'),
            "requested_changes": [
                "Add more quantified leadership achievements",
                "Highlight team size and management responsibilities"
            ]
        }
        
        # Process with user feedback
        feedback_result = await orchestrator.process_individual_item_with_feedback(
            item=experience_item,
            job_description=ai_engineer_job_description,
            session_id="test-feedback-integration",
            item_type=ContentType.EXPERIENCE,
            user_feedback=user_feedback
        )
        
        # Assertions: Feedback integration
        assert feedback_result.get('status') == ProcessingStatus.COMPLETED
        feedback_bullets = feedback_result.get('generated_content', {}).get('tailored_bullets', [])
        
        # Content should be different after feedback
        assert feedback_bullets != initial_bullets, "Content should change after user feedback"
        
        # Should contain leadership-related keywords
        feedback_text = ' '.join(feedback_bullets).lower()
        leadership_keywords = ['led', 'team', 'managed', 'leadership', 'engineers']
        leadership_matches = sum(1 for keyword in leadership_keywords if keyword in feedback_text)
        assert leadership_matches >= 2, f"Expected leadership emphasis, found {leadership_matches} matches"
        
        print("✅ User feedback integration test passed")
    
    @pytest.mark.asyncio
    async def test_batch_individual_processing(
        self,
        orchestrator_with_rate_limiting,
        sample_experience_items,
        ai_engineer_job_description,
        performance_timer
    ):
        """Test batch processing of individual items with rate limit management."""
        orchestrator = orchestrator_with_rate_limiting
        performance_timer.start()
        
        # Create content items from experience items
        content_items = [
            ContentItem(
                content_type=ContentType.EXPERIENCE,
                original_content=json.dumps({
                    "title": item.title,
                    "company": item.company,
                    "duration": item.duration,
                    "responsibilities": item.responsibilities,
                    "technologies": item.technologies
                }),
                metadata=ProcessingMetadata(
                    item_id=f"exp-{i}",
                    status=ProcessingStatus.PENDING
                )
            )
            for i, item in enumerate(sample_experience_items)
        ]
        
        # Process batch with rate limiting
        batch_results = await orchestrator.process_item_batch(
            items=content_items,
            job_description=ai_engineer_job_description,
            session_id="test-batch-individual",
            max_concurrent=2  # Limit concurrency for rate limiting
        )
        
        performance_timer.stop()
        
        # Assertions: Batch processing
        assert len(batch_results) == len(content_items)
        
        # All items should be processed successfully
        successful_items = [r for r in batch_results if r.get('status') == ProcessingStatus.COMPLETED]
        assert len(successful_items) == len(content_items), "All items should be processed successfully"
        
        # Rate limiting should keep total time reasonable
        assert performance_timer.elapsed < 30.0, f"Batch processing took {performance_timer.elapsed:.2f}s, expected < 30s"
        
        # Verify content quality across all items
        for i, result in enumerate(batch_results):
            tailored_content = result.get('generated_content', {})
            bullets = tailored_content.get('tailored_bullets', [])
            
            assert len(bullets) >= 3, f"Item {i} should have at least 3 bullets"
            
            # Each item should be tailored to the job
            bullets_text = ' '.join(bullets).lower()
            assert any(keyword in bullets_text for keyword in ['ml', 'data', 'python', 'model']), \
                f"Item {i} should contain job-relevant keywords"
        
        print(f"✅ Batch individual processing test completed in {performance_timer.elapsed:.2f}s")
        print(f"   Processed {len(successful_items)} items successfully")
    
    @pytest.mark.asyncio
    async def test_rate_limit_recovery(
        self,
        orchestrator_with_rate_limiting,
        sample_experience_items,
        ai_engineer_job_description
    ):
        """Test rate limit detection and recovery mechanisms."""
        orchestrator = orchestrator_with_rate_limiting
        
        # Mock rate limiter to simulate rate limit hit
        with patch.object(orchestrator.rate_limiter, 'check_rate_limit') as mock_check:
            with patch.object(orchestrator.rate_limiter, 'wait_for_availability') as mock_wait:
                
                # Simulate rate limit on 3rd call
                mock_check.side_effect = [True, True, False, True, True]
                mock_wait.return_value = asyncio.sleep(0.1)  # Short wait for testing
                
                experience_item = sample_experience_items[0]
                
                # This should trigger rate limit handling
                result = await orchestrator.process_individual_item(
                    item=experience_item,
                    job_description=ai_engineer_job_description,
                    session_id="test-rate-limit-recovery",
                    item_type=ContentType.EXPERIENCE
                )
                
                # Should still complete successfully after rate limit recovery
                assert result.get('status') == ProcessingStatus.COMPLETED
                
                # Verify rate limit methods were called
                assert mock_check.called, "Rate limit check should be called"
                
                print("✅ Rate limit recovery test passed")