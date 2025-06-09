"""End-to-End Test 3: Error Recovery

Tests error handling and recovery workflows with invalid job descriptions,
API failures, graceful degradation, and retry success mechanisms.
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
    JobDescriptionData, ExperienceItem, ContentItem
)
from src.services.error_recovery import ErrorRecoveryService
from src.services.session_manager import SessionManager
from src.services.progress_tracker import ProgressTracker
from src.config.logging_config import get_structured_logger


class APIError(Exception):
    """Custom API error for testing."""
    pass


class RateLimitError(Exception):
    """Custom rate limit error for testing."""
    pass


class TimeoutError(Exception):
    """Custom timeout error for testing."""
    pass


@pytest.mark.e2e
@pytest.mark.slow
class TestErrorRecovery:
    """End-to-end tests for error recovery workflows."""
    
    @pytest.fixture
    def invalid_job_descriptions(self) -> List[Dict[str, Any]]:
        """Provide various invalid job description scenarios."""
        return [
            {
                "name": "empty_description",
                "data": JobDescriptionData(
                    title="",
                    company="",
                    description="",
                    requirements=[],
                    responsibilities=[],
                    skills=[]
                )
            },
            {
                "name": "malformed_content",
                "data": JobDescriptionData(
                    title="Software Engineer",
                    company="TechCorp",
                    description="<script>alert('xss')</script>Invalid content with HTML",
                    requirements=["Invalid requirement with special chars: @#$%^&*()"],
                    responsibilities=["Malformed responsibility"],
                    skills=[""]
                )
            },
            {
                "name": "extremely_long_content",
                "data": JobDescriptionData(
                    title="AI Engineer",
                    company="TechCorp",
                    description="A" * 10000,  # Extremely long description
                    requirements=["Requirement " + "A" * 1000] * 50,  # Many long requirements
                    responsibilities=["Responsibility " + "B" * 1000] * 50,
                    skills=["Skill" + str(i) for i in range(200)]  # Too many skills
                )
            }
        ]
    
    @pytest.fixture
    def valid_base_cv(self) -> str:
        """Provide a valid base CV for error recovery testing."""
        return """
        # Jane Doe
        ## Professional Experience
        
        ### Software Engineer | TechCorp | 2020-2023
        - Developed web applications using Python and React
        - Implemented RESTful APIs for mobile clients
        - Worked with PostgreSQL databases
        
        ## Education
        
        ### Bachelor of Computer Science | University | 2020
        - Relevant coursework in software engineering
        
        ## Skills
        - Python, JavaScript, SQL
        - React, Flask, PostgreSQL
        """
    
    @pytest.fixture
    def failing_llm_client(self):
        """Provide a mock LLM client that simulates various API failures."""
        mock_client = Mock()
        mock_client.generate_content = AsyncMock()
        
        # Track call attempts for retry testing
        call_attempts = {'count': 0, 'failure_mode': 'none'}
        
        async def failing_generate(*args, **kwargs):
            call_attempts['count'] += 1
            
            # Different failure modes based on test configuration
            if call_attempts['failure_mode'] == 'rate_limit':
                if call_attempts['count'] <= 2:
                    raise RateLimitError("Rate limit exceeded")
                else:
                    # Success after retries
                    return {
                        "content": json.dumps({"tailored_bullets": ["Recovered content"]}),
                        "tokens_used": 50,
                        "model": "gpt-4"
                    }
            
            elif call_attempts['failure_mode'] == 'timeout':
                if call_attempts['count'] <= 1:
                    raise TimeoutError("Request timeout")
                else:
                    return {
                        "content": json.dumps({"tailored_bullets": ["Recovered after timeout"]}),
                        "tokens_used": 50,
                        "model": "gpt-4"
                    }
            
            elif call_attempts['failure_mode'] == 'authentication':
                raise APIError("Authentication failed")
            
            elif call_attempts['failure_mode'] == 'permanent_failure':
                raise APIError("Permanent API failure")
            
            else:
                # Normal operation
                return {
                    "content": json.dumps({"tailored_bullets": ["Normal content"]}),
                    "tokens_used": 50,
                    "model": "gpt-4"
                }
        
        mock_client.generate_content.side_effect = failing_generate
        mock_client._call_attempts = call_attempts  # Expose for test control
        
        return mock_client
    
    @pytest.fixture
    def orchestrator_with_error_recovery(self, failing_llm_client, temp_dir):
        """Provide an orchestrator configured for error recovery testing."""
        # Mock error recovery service
        mock_error_recovery = Mock(spec=ErrorRecoveryService)
        mock_error_recovery.handle_error = AsyncMock()
        mock_error_recovery.should_retry = Mock(return_value=True)
        mock_error_recovery.get_retry_delay = Mock(return_value=0.1)
        mock_error_recovery.get_max_retries = Mock(return_value=3)
        
        # Mock session manager with state preservation
        mock_session_manager = Mock(spec=SessionManager)
        mock_session_manager.save_session_state = AsyncMock()
        mock_session_manager.restore_session_state = AsyncMock()
        
        # Mock progress tracker
        mock_progress_tracker = Mock(spec=ProgressTracker)
        
        orchestrator = EnhancedOrchestrator(
            llm_client=failing_llm_client,
            error_recovery=mock_error_recovery,
            session_manager=mock_session_manager,
            progress_tracker=mock_progress_tracker
        )
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_invalid_job_description_handling(
        self,
        orchestrator_with_error_recovery,
        invalid_job_descriptions,
        valid_base_cv
    ):
        """Test E2E Test 3: Error Recovery with invalid job descriptions.
        
        Input: Invalid job description or API failures
        Process: Error handling and recovery workflows
        Output: Graceful degradation or retry success
        Assertions: User notification, state preservation
        """
        orchestrator = orchestrator_with_error_recovery
        
        for invalid_job in invalid_job_descriptions:
            job_name = invalid_job['name']
            job_data = invalid_job['data']
            
            print(f"Testing invalid job description: {job_name}")
            
            try:
                result = await orchestrator.process_complete_cv(
                    job_description=job_data,
                    base_cv_content=valid_base_cv,
                    session_id=f"test-invalid-{job_name}"
                )
                
                # Should handle gracefully, not crash
                assert result is not None
                
                # Check for graceful degradation
                if result.get('status') == 'failed':
                    # Should provide meaningful error information
                    error_info = result.get('error_info', {})
                    assert 'error_type' in error_info
                    assert 'user_message' in error_info
                    assert error_info['user_message'] != "", "Should provide user-friendly error message"
                    
                    # Should preserve session state for recovery
                    assert 'session_id' in result
                    assert result['session_id'] is not None
                    
                elif result.get('status') == 'partial_success':
                    # Should indicate what was completed and what failed
                    assert 'completed_sections' in result
                    assert 'failed_sections' in result
                    
                    # Should still provide some usable content
                    generated_content = result.get('generated_content', {})
                    assert len(generated_content) > 0, "Should provide some content even with partial success"
                
                print(f"✅ Handled {job_name} gracefully")
                
            except Exception as e:
                pytest.fail(f"Failed to handle invalid job description {job_name} gracefully: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_api_failure_retry_mechanisms(
        self,
        orchestrator_with_error_recovery,
        valid_base_cv
    ):
        """Test retry mechanisms for various API failures."""
        orchestrator = orchestrator_with_error_recovery
        
        # Valid job description for testing
        job_data = JobDescriptionData(
            title="Software Engineer",
            company="TechCorp",
            description="We are looking for a software engineer.",
            requirements=["Python experience"],
            responsibilities=["Develop software"],
            skills=["Python", "SQL"]
        )
        
        # Test rate limit recovery
        orchestrator.llm_client._call_attempts['failure_mode'] = 'rate_limit'
        orchestrator.llm_client._call_attempts['count'] = 0
        
        result = await orchestrator.process_complete_cv(
            job_description=job_data,
            base_cv_content=valid_base_cv,
            session_id="test-rate-limit-recovery"
        )
        
        # Should succeed after retries
        assert result.get('status') in ['completed', 'partial_success']
        assert orchestrator.llm_client._call_attempts['count'] > 1, "Should have retried"
        
        print("✅ Rate limit recovery test passed")
        
        # Test timeout recovery
        orchestrator.llm_client._call_attempts['failure_mode'] = 'timeout'
        orchestrator.llm_client._call_attempts['count'] = 0
        
        result = await orchestrator.process_complete_cv(
            job_description=job_data,
            base_cv_content=valid_base_cv,
            session_id="test-timeout-recovery"
        )
        
        # Should succeed after timeout retry
        assert result.get('status') in ['completed', 'partial_success']
        assert orchestrator.llm_client._call_attempts['count'] > 1, "Should have retried after timeout"
        
        print("✅ Timeout recovery test passed")
    
    @pytest.mark.asyncio
    async def test_permanent_failure_graceful_degradation(
        self,
        orchestrator_with_error_recovery,
        valid_base_cv
    ):
        """Test graceful degradation when permanent failures occur."""
        orchestrator = orchestrator_with_error_recovery
        
        job_data = JobDescriptionData(
            title="Software Engineer",
            company="TechCorp",
            description="We are looking for a software engineer.",
            requirements=["Python experience"],
            responsibilities=["Develop software"],
            skills=["Python", "SQL"]
        )
        
        # Test permanent authentication failure
        orchestrator.llm_client._call_attempts['failure_mode'] = 'authentication'
        
        result = await orchestrator.process_complete_cv(
            job_description=job_data,
            base_cv_content=valid_base_cv,
            session_id="test-auth-failure"
        )
        
        # Should fail gracefully
        assert result.get('status') == 'failed'
        
        # Should provide meaningful error information
        error_info = result.get('error_info', {})
        assert 'error_type' in error_info
        assert error_info['error_type'] == 'authentication_error'
        assert 'user_message' in error_info
        assert 'suggested_actions' in error_info
        
        # Should preserve original CV content as fallback
        fallback_content = result.get('fallback_content')
        assert fallback_content is not None
        assert valid_base_cv in str(fallback_content)
        
        print("✅ Permanent failure graceful degradation test passed")
    
    @pytest.mark.asyncio
    async def test_state_preservation_during_errors(
        self,
        orchestrator_with_error_recovery,
        valid_base_cv
    ):
        """Test that session state is preserved during error scenarios."""
        orchestrator = orchestrator_with_error_recovery
        
        job_data = JobDescriptionData(
            title="Software Engineer",
            company="TechCorp",
            description="We are looking for a software engineer.",
            requirements=["Python experience"],
            responsibilities=["Develop software"],
            skills=["Python", "SQL"]
        )
        
        session_id = "test-state-preservation"
        
        # Configure for failure after partial processing
        orchestrator.llm_client._call_attempts['failure_mode'] = 'permanent_failure'
        
        result = await orchestrator.process_complete_cv(
            job_description=job_data,
            base_cv_content=valid_base_cv,
            session_id=session_id
        )
        
        # Should have attempted to save session state
        orchestrator.session_manager.save_session_state.assert_called()
        
        # Verify session state preservation call
        save_calls = orchestrator.session_manager.save_session_state.call_args_list
        assert len(save_calls) > 0, "Should have saved session state"
        
        # Check that session ID is preserved in result
        assert result.get('session_id') == session_id
        
        # Should indicate that state was preserved for recovery
        assert result.get('state_preserved') is True
        
        print("✅ State preservation test passed")
    
    @pytest.mark.asyncio
    async def test_user_notification_during_errors(
        self,
        orchestrator_with_error_recovery,
        valid_base_cv
    ):
        """Test that users receive appropriate notifications during errors."""
        orchestrator = orchestrator_with_error_recovery
        
        # Capture progress notifications
        notifications = []
        
        def capture_notification(notification):
            notifications.append(notification)
        
        orchestrator.progress_callback = capture_notification
        
        job_data = JobDescriptionData(
            title="Software Engineer",
            company="TechCorp",
            description="We are looking for a software engineer.",
            requirements=["Python experience"],
            responsibilities=["Develop software"],
            skills=["Python", "SQL"]
        )
        
        # Configure for rate limit errors
        orchestrator.llm_client._call_attempts['failure_mode'] = 'rate_limit'
        orchestrator.llm_client._call_attempts['count'] = 0
        
        result = await orchestrator.process_complete_cv(
            job_description=job_data,
            base_cv_content=valid_base_cv,
            session_id="test-user-notifications"
        )
        
        # Should have sent progress notifications
        assert len(notifications) > 0, "Should have sent progress notifications"
        
        # Check for error-related notifications
        error_notifications = [n for n in notifications if n.get('type') == 'error']
        retry_notifications = [n for n in notifications if n.get('type') == 'retry']
        
        # Should notify about errors and retries
        assert len(error_notifications) > 0, "Should have error notifications"
        assert len(retry_notifications) > 0, "Should have retry notifications"
        
        # Notifications should be user-friendly
        for notification in error_notifications:
            assert 'message' in notification
            assert notification['message'] != "", "Error messages should not be empty"
            assert 'timestamp' in notification
        
        print(f"✅ User notification test passed ({len(notifications)} notifications sent)")
    
    @pytest.mark.asyncio
    async def test_partial_success_recovery(
        self,
        orchestrator_with_error_recovery,
        valid_base_cv
    ):
        """Test recovery scenarios where some sections succeed and others fail."""
        orchestrator = orchestrator_with_error_recovery
        
        job_data = JobDescriptionData(
            title="Software Engineer",
            company="TechCorp",
            description="We are looking for a software engineer.",
            requirements=["Python experience"],
            responsibilities=["Develop software"],
            skills=["Python", "SQL"]
        )
        
        # Mock partial success scenario
        with patch.object(orchestrator, 'process_cv_section') as mock_process:
            # First section succeeds, second fails, third succeeds
            mock_process.side_effect = [
                {'status': 'completed', 'content': {'skills': ['Python', 'SQL']}},
                APIError("Section processing failed"),
                {'status': 'completed', 'content': {'projects': ['Web App']}}
            ]
            
            result = await orchestrator.process_complete_cv(
                job_description=job_data,
                base_cv_content=valid_base_cv,
                session_id="test-partial-success"
            )
            
            # Should indicate partial success
            assert result.get('status') == 'partial_success'
            
            # Should list completed and failed sections
            assert 'completed_sections' in result
            assert 'failed_sections' in result
            
            completed_sections = result['completed_sections']
            failed_sections = result['failed_sections']
            
            assert len(completed_sections) == 2, "Should have 2 completed sections"
            assert len(failed_sections) == 1, "Should have 1 failed section"
            
            # Should provide partial content
            generated_content = result.get('generated_content', {})
            assert 'skills' in generated_content
            assert 'projects' in generated_content
            
            print("✅ Partial success recovery test passed")
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(
        self,
        orchestrator_with_error_recovery,
        valid_base_cv,
        performance_timer
    ):
        """Test that error recovery doesn't cause excessive delays."""
        orchestrator = orchestrator_with_error_recovery
        performance_timer.start()
        
        job_data = JobDescriptionData(
            title="Software Engineer",
            company="TechCorp",
            description="We are looking for a software engineer.",
            requirements=["Python experience"],
            responsibilities=["Develop software"],
            skills=["Python", "SQL"]
        )
        
        # Configure for timeout with recovery
        orchestrator.llm_client._call_attempts['failure_mode'] = 'timeout'
        orchestrator.llm_client._call_attempts['count'] = 0
        
        result = await orchestrator.process_complete_cv(
            job_description=job_data,
            base_cv_content=valid_base_cv,
            session_id="test-error-performance"
        )
        
        performance_timer.stop()
        
        # Should complete within reasonable time even with retries
        assert performance_timer.elapsed < 10.0, f"Error recovery took {performance_timer.elapsed:.2f}s, expected < 10s"
        
        # Should still succeed
        assert result.get('status') in ['completed', 'partial_success']
        
        print(f"✅ Error recovery performance test passed ({performance_timer.elapsed:.2f}s)")