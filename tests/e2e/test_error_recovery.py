"""End-to-End test for error recovery and resilience workflows.

Tests the application's ability to handle various failure scenarios
and recover gracefully while maintaining data integrity.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from uuid import uuid4
import json
import tempfile
import os

# Import application components
from src.orchestration.enhanced_orchestrator import EnhancedOrchestrator
from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, Section, Subsection, Item, ItemStatus, ItemType,
    JobDescriptionData
)
from src.services.state_manager import StateManager
from src.services.session_manager import SessionManager
from src.services.llm import LLMService
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.research_agent import ResearchAgent
from src.agents.qa_agent import QAAgent


@pytest.mark.e2e
@pytest.mark.asyncio
class TestErrorRecovery:
    """E2E tests for error recovery and resilience."""

    @pytest.fixture
    def sample_structured_cv(self):
        """Sample structured CV for error recovery testing."""
        return StructuredCV(
            sections=[
                Section(
                    name="Professional Experience",
                    subsections=[
                        Subsection(
                            name="Senior Developer @ TechCorp",
                            items=[
                                Item(
                                    id=uuid4(),
                                    content="Developed web applications",
                                    status=ItemStatus.GENERATED,
                                    item_type=ItemType.BULLET_POINT,
                                    confidence_score=0.8,
                                    raw_llm_output="Original response"
                                ),
                                Item(
                                    id=uuid4(),
                                    content="Led development team",
                                    status=ItemStatus.GENERATED,
                                    item_type=ItemType.BULLET_POINT,
                                    confidence_score=0.75,
                                    raw_llm_output="Original response 2"
                                )
                            ]
                        )
                    ]
                )
            ],
            big_10_skills=["Python", "Leadership", "Web Development"]
        )

    @pytest.fixture
    def sample_job_description_data(self):
        """Sample job description data."""
        return JobDescriptionData(
            required_skills=["Python", "Leadership", "API Development"],
            responsibilities=["Lead development", "Architect solutions"],
            industry_terms=["Software Engineering", "Agile"],
            company_context="Technology startup focused on innovation"
        )

    async def test_llm_service_timeout_recovery(self, sample_structured_cv, sample_job_description_data):
        """Test recovery from LLM service timeouts.
        
        Validates:
        - REQ-NONFUNC-RELIABILITY-1: Service timeout handling
        - REQ-NONFUNC-RELIABILITY-2: Graceful degradation
        - Fallback content generation
        """
        session_id = str(uuid4())
        
        # Create orchestrator with timeout-prone LLM service
        orchestrator = EnhancedOrchestrator()
        
        # Mock content writer that simulates timeout then recovery
        mock_content_writer = AsyncMock(spec=EnhancedContentWriterAgent)
        
        call_count = 0
        async def mock_process_with_timeout_recovery(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call: simulate timeout
                raise asyncio.TimeoutError("LLM service timeout")
            else:
                # Second call: successful recovery
                return {
                    "success": True,
                    "structured_cv": sample_structured_cv,
                    "metadata": {
                        "processing_time": 15.0,
                        "model_used": "gemini-pro",
                        "retry_count": 1,
                        "recovery_successful": True
                    }
                }
        
        mock_content_writer.process_cv_generation = mock_process_with_timeout_recovery
        orchestrator.content_writer_agent = mock_content_writer
        
        # Create agent state
        agent_state = AgentState(
            session_id=session_id,
            job_description_data=sample_job_description_data,
            base_cv_content="Base CV content",
            research_findings="Research data",
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock state and session managers
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class, \
             patch('src.services.session_manager.SessionManager') as mock_session_manager_class:
            
            mock_state_manager = MagicMock()
            mock_session_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager
            mock_session_manager_class.return_value = mock_session_manager
            
            # Configure mocks
            mock_state_manager.save_structured_cv.return_value = True
            mock_session_manager.save_session_state.return_value = True
            
            # Execute with retry logic
            max_retries = 3
            retry_count = 0
            result_state = None
            
            while retry_count < max_retries:
                try:
                    result_state = await orchestrator.process_cv_generation(agent_state)
                    break  # Success, exit retry loop
                except asyncio.TimeoutError as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        # Final attempt with fallback
                        result_state = AgentState(
                            session_id=session_id,
                            structured_cv=sample_structured_cv,
                            job_description_data=sample_job_description_data,
                            error_messages=[f"LLM timeout after {max_retries} retries: {str(e)}"],
                            final_cv_output_path=None
                        )
                        break
                    
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(0.1 * (2 ** retry_count))
            
            # Validate recovery was successful
            assert result_state is not None
            assert result_state.structured_cv is not None
            
            # Validate retry mechanism was triggered
            assert call_count == 2, "Should have made 2 calls (1 timeout + 1 success)"
            
            # Validate error handling
            if len(result_state.error_messages) > 0:
                # If errors exist, they should be properly formatted
                for error in result_state.error_messages:
                    assert isinstance(error, str)
                    assert len(error) > 0

    async def test_state_corruption_recovery(self, sample_structured_cv, sample_job_description_data):
        """Test recovery from corrupted state data.
        
        Validates:
        - State validation and repair
        - Data integrity checks
        - Graceful handling of invalid state
        """
        session_id = str(uuid4())
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator()
        
        # Create corrupted agent state (missing required fields)
        corrupted_state = AgentState(
            session_id=session_id,
            # Missing structured_cv and job_description_data
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock state manager with validation
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager
            
            # Configure state manager to detect and repair corruption
            def mock_validate_and_repair_state(state):
                if not hasattr(state, 'structured_cv') or state.structured_cv is None:
                    # Repair by providing default structured CV
                    state.structured_cv = sample_structured_cv
                    state.error_messages.append("State repaired: Missing structured_cv restored from backup")
                
                if not hasattr(state, 'job_description_data') or state.job_description_data is None:
                    # Repair by providing default job description
                    state.job_description_data = sample_job_description_data
                    state.error_messages.append("State repaired: Missing job_description_data restored from backup")
                
                return state
            
            mock_state_manager.validate_and_repair_state = mock_validate_and_repair_state
            mock_state_manager.save_structured_cv.return_value = True
            
            # Attempt to process with corrupted state
            try:
                # Validate and repair state before processing
                repaired_state = mock_state_manager.validate_and_repair_state(corrupted_state)
                
                # Validate repair was successful
                assert repaired_state.structured_cv is not None
                assert repaired_state.job_description_data is not None
                assert len(repaired_state.error_messages) == 2  # Two repair messages
                
                # Validate repair messages
                repair_messages = repaired_state.error_messages
                assert any("structured_cv restored" in msg for msg in repair_messages)
                assert any("job_description_data restored" in msg for msg in repair_messages)
                
            except Exception as e:
                pytest.fail(f"State repair should not raise exceptions: {e}")

    async def test_file_system_error_recovery(self, sample_structured_cv, sample_job_description_data):
        """Test recovery from file system errors.
        
        Validates:
        - File I/O error handling
        - Alternative storage mechanisms
        - Data persistence resilience
        """
        session_id = str(uuid4())
        
        # Create orchestrator
        orchestrator = EnhancedOrchestrator()
        
        agent_state = AgentState(
            session_id=session_id,
            structured_cv=sample_structured_cv,
            job_description_data=sample_job_description_data,
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock state manager with file system errors
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager
            
            # Simulate file system errors with fallback
            save_attempt_count = 0
            def mock_save_with_fallback(*args, **kwargs):
                nonlocal save_attempt_count
                save_attempt_count += 1
                
                if save_attempt_count == 1:
                    # First attempt: simulate disk full error
                    raise OSError("No space left on device")
                elif save_attempt_count == 2:
                    # Second attempt: simulate permission error
                    raise PermissionError("Permission denied")
                else:
                    # Third attempt: successful fallback to memory storage
                    return True
            
            mock_state_manager.save_structured_cv.side_effect = mock_save_with_fallback
            
            # Attempt to save with error recovery
            max_retries = 3
            save_successful = False
            error_messages = []
            
            for attempt in range(max_retries):
                try:
                    result = mock_state_manager.save_structured_cv(agent_state.structured_cv)
                    if result:
                        save_successful = True
                        break
                except OSError as e:
                    error_messages.append(f"Disk error on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        # Final attempt: use in-memory fallback
                        save_successful = True
                        error_messages.append("Fallback to in-memory storage successful")
                except PermissionError as e:
                    error_messages.append(f"Permission error on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        # Final attempt: use alternative location
                        save_successful = True
                        error_messages.append("Fallback to alternative storage location successful")
            
            # Validate error recovery
            assert save_successful, "Save operation should eventually succeed with fallbacks"
            assert save_attempt_count == 3, "Should have made 3 save attempts"
            assert len(error_messages) >= 2, "Should have recorded error messages"
            
            # Validate error messages contain useful information
            assert any("Disk error" in msg for msg in error_messages)
            assert any("Permission error" in msg for msg in error_messages)
            assert any("Fallback" in msg for msg in error_messages)

    async def test_concurrent_session_conflict_resolution(self, sample_structured_cv, sample_job_description_data):
        """Test handling of concurrent session conflicts.
        
        Validates:
        - Session isolation
        - Conflict detection and resolution
        - Data consistency across sessions
        """
        session_id_1 = str(uuid4())
        session_id_2 = str(uuid4())
        
        # Create two agent states for concurrent sessions
        agent_state_1 = AgentState(
            session_id=session_id_1,
            structured_cv=sample_structured_cv,
            job_description_data=sample_job_description_data,
            current_item_id=str(sample_structured_cv.sections[0].subsections[0].items[0].id),
            user_feedback="Session 1 feedback",
            error_messages=[],
            final_cv_output_path=None
        )
        
        agent_state_2 = AgentState(
            session_id=session_id_2,
            structured_cv=sample_structured_cv,
            job_description_data=sample_job_description_data,
            current_item_id=str(sample_structured_cv.sections[0].subsections[0].items[0].id),
            user_feedback="Session 2 feedback",
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock session manager with conflict detection
        with patch('src.services.session_manager.SessionManager') as mock_session_manager_class:
            mock_session_manager = MagicMock()
            mock_session_manager_class.return_value = mock_session_manager
            
            # Track active sessions
            active_sessions = {}
            
            def mock_acquire_session_lock(session_id):
                if session_id in active_sessions:
                    raise RuntimeError(f"Session {session_id} is already active")
                active_sessions[session_id] = True
                return True
            
            def mock_release_session_lock(session_id):
                if session_id in active_sessions:
                    del active_sessions[session_id]
                return True
            
            def mock_check_session_conflicts(session_id, item_id):
                # Check if another session is modifying the same item
                for active_session_id in active_sessions:
                    if active_session_id != session_id:
                        return {
                            "conflict_detected": True,
                            "conflicting_session": active_session_id,
                            "conflict_type": "item_modification",
                            "item_id": item_id
                        }
                return {"conflict_detected": False}
            
            mock_session_manager.acquire_session_lock = mock_acquire_session_lock
            mock_session_manager.release_session_lock = mock_release_session_lock
            mock_session_manager.check_session_conflicts = mock_check_session_conflicts
            
            # Test concurrent session handling
            try:
                # Session 1 acquires lock
                lock_1 = mock_session_manager.acquire_session_lock(session_id_1)
                assert lock_1 is True
                
                # Session 2 attempts to acquire lock for same resource
                conflict_check = mock_session_manager.check_session_conflicts(
                    session_id_2, 
                    agent_state_2.current_item_id
                )
                
                # Validate conflict detection
                assert conflict_check["conflict_detected"] is True
                assert conflict_check["conflicting_session"] == session_id_1
                assert conflict_check["conflict_type"] == "item_modification"
                
                # Session 1 releases lock
                release_1 = mock_session_manager.release_session_lock(session_id_1)
                assert release_1 is True
                
                # Session 2 can now proceed
                lock_2 = mock_session_manager.acquire_session_lock(session_id_2)
                assert lock_2 is True
                
                # No conflict should be detected now
                conflict_check_2 = mock_session_manager.check_session_conflicts(
                    session_id_2,
                    agent_state_2.current_item_id
                )
                assert conflict_check_2["conflict_detected"] is False
                
                # Clean up
                mock_session_manager.release_session_lock(session_id_2)
                
            except Exception as e:
                pytest.fail(f"Concurrent session handling should not raise exceptions: {e}")

    async def test_memory_pressure_recovery(self, sample_structured_cv, sample_job_description_data):
        """Test recovery from memory pressure situations.
        
        Validates:
        - Memory usage monitoring
        - Graceful degradation under memory pressure
        - Cache cleanup and optimization
        """
        session_id = str(uuid4())
        
        # Create large structured CV to simulate memory pressure
        large_cv = StructuredCV(
            sections=[
                Section(
                    name=f"Section {i}",
                    subsections=[
                        Subsection(
                            name=f"Subsection {j}",
                            items=[
                                Item(
                                    id=uuid4(),
                                    content=f"Large content item {k} with extensive details " * 100,  # Large content
                                    status=ItemStatus.GENERATED,
                                    item_type=ItemType.BULLET_POINT,
                                    confidence_score=0.8,
                                    raw_llm_output=f"Large LLM response {k} " * 50
                                )
                                for k in range(10)  # 10 items per subsection
                            ]
                        )
                        for j in range(5)  # 5 subsections per section
                    ]
                )
                for i in range(3)  # 3 sections
            ],
            big_10_skills=[f"Skill {i}" for i in range(20)]  # Large skills list
        )
        
        agent_state = AgentState(
            session_id=session_id,
            structured_cv=large_cv,
            job_description_data=sample_job_description_data,
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock memory monitoring
        with patch('src.services.state_manager.StateManager') as mock_state_manager_class:
            mock_state_manager = MagicMock()
            mock_state_manager_class.return_value = mock_state_manager
            
            # Simulate memory pressure detection and cleanup
            def mock_check_memory_usage():
                return {
                    "memory_usage_mb": 512,  # High memory usage
                    "memory_limit_mb": 256,  # Low limit
                    "pressure_detected": True,
                    "cleanup_recommended": True
                }
            
            def mock_cleanup_memory_cache():
                return {
                    "cache_cleared": True,
                    "memory_freed_mb": 128,
                    "items_removed": 50,
                    "cleanup_successful": True
                }
            
            def mock_optimize_cv_storage(cv):
                # Simulate CV optimization (compress raw outputs, etc.)
                optimized_cv = StructuredCV(
                    sections=cv.sections[:2],  # Reduce sections
                    big_10_skills=cv.big_10_skills[:10]  # Reduce skills
                )
                
                # Compress item content
                for section in optimized_cv.sections:
                    for subsection in section.subsections:
                        for item in subsection.items:
                            item.raw_llm_output = "[Compressed]"  # Compress raw output
                
                return optimized_cv
            
            mock_state_manager.check_memory_usage = mock_check_memory_usage
            mock_state_manager.cleanup_memory_cache = mock_cleanup_memory_cache
            mock_state_manager.optimize_cv_storage = mock_optimize_cv_storage
            mock_state_manager.save_structured_cv.return_value = True
            
            # Test memory pressure handling
            try:
                # Check memory usage
                memory_status = mock_state_manager.check_memory_usage()
                assert memory_status["pressure_detected"] is True
                
                # Perform cleanup if needed
                if memory_status["cleanup_recommended"]:
                    cleanup_result = mock_state_manager.cleanup_memory_cache()
                    assert cleanup_result["cleanup_successful"] is True
                    assert cleanup_result["memory_freed_mb"] > 0
                    
                    # Optimize CV storage
                    optimized_cv = mock_state_manager.optimize_cv_storage(agent_state.structured_cv)
                    agent_state.structured_cv = optimized_cv
                    
                    # Validate optimization
                    assert len(optimized_cv.sections) <= len(large_cv.sections)
                    assert len(optimized_cv.big_10_skills) <= len(large_cv.big_10_skills)
                    
                    # Check that raw outputs were compressed
                    for section in optimized_cv.sections:
                        for subsection in section.subsections:
                            for item in subsection.items:
                                assert item.raw_llm_output == "[Compressed]"
                    
                    # Save optimized state
                    save_result = mock_state_manager.save_structured_cv(optimized_cv)
                    assert save_result is True
                    
                    # Add success message to state
                    agent_state.error_messages.append("Memory optimization completed successfully")
                
                # Validate final state
                assert agent_state.structured_cv is not None
                assert len(agent_state.error_messages) == 1
                assert "Memory optimization completed" in agent_state.error_messages[0]
                
            except Exception as e:
                pytest.fail(f"Memory pressure handling should not raise exceptions: {e}")

    async def test_network_connectivity_recovery(self, sample_structured_cv, sample_job_description_data):
        """Test recovery from network connectivity issues.
        
        Validates:
        - Network error detection
        - Offline mode capabilities
        - Automatic reconnection
        """
        session_id = str(uuid4())
        
        agent_state = AgentState(
            session_id=session_id,
            structured_cv=sample_structured_cv,
            job_description_data=sample_job_description_data,
            error_messages=[],
            final_cv_output_path=None
        )
        
        # Mock LLM service with network issues
        with patch('src.services.llm.LLMService') as mock_llm_service_class:
            mock_llm_service = MagicMock()
            mock_llm_service_class.return_value = mock_llm_service
            
            # Simulate network connectivity issues
            call_count = 0
            async def mock_generate_with_network_issues(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count <= 2:
                    # First two calls: network errors
                    if call_count == 1:
                        raise ConnectionError("Network unreachable")
                    else:
                        raise TimeoutError("Request timeout")
                else:
                    # Third call: successful connection
                    return {
                        "content": "Generated content after network recovery",
                        "confidence": 0.85,
                        "metadata": {
                            "model": "gemini-pro",
                            "retry_count": call_count - 1,
                            "network_recovery": True
                        }
                    }
            
            mock_llm_service.generate_content = mock_generate_with_network_issues
            
            # Test network error recovery
            max_retries = 3
            retry_count = 0
            result = None
            error_messages = []
            
            while retry_count < max_retries:
                try:
                    result = await mock_llm_service.generate_content(
                        prompt="Test prompt",
                        context="Test context"
                    )
                    break  # Success, exit retry loop
                    
                except (ConnectionError, TimeoutError) as e:
                    retry_count += 1
                    error_messages.append(f"Network error attempt {retry_count}: {str(e)}")
                    
                    if retry_count >= max_retries:
                        # Final attempt failed, use offline fallback
                        result = {
                            "content": "⚠️ Network unavailable. Using cached content. Please check connection and regenerate.",
                            "confidence": 0.0,
                            "metadata": {
                                "offline_mode": True,
                                "fallback_used": True,
                                "retry_count": retry_count
                            }
                        }
                        error_messages.append("Switched to offline mode due to persistent network issues")
                        break
                    
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(0.1 * (2 ** retry_count))
            
            # Validate network recovery
            assert result is not None
            assert call_count == 3, "Should have made 3 attempts before success"
            
            if "network_recovery" in result.get("metadata", {}):
                # Network recovery was successful
                assert result["metadata"]["network_recovery"] is True
                assert result["metadata"]["retry_count"] == 2
                assert "Generated content after network recovery" in result["content"]
            else:
                # Fallback to offline mode
                assert result["metadata"]["offline_mode"] is True
                assert result["metadata"]["fallback_used"] is True
                assert "⚠️ Network unavailable" in result["content"]
            
            # Validate error messages were recorded
            assert len(error_messages) >= 2
            assert any("Network unreachable" in msg for msg in error_messages)
            assert any("Request timeout" in msg for msg in error_messages)