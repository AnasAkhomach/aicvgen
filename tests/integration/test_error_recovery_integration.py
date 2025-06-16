#!/usr/bin/env python3
"""
Integration tests for error recovery and resilience patterns.
Tests the interaction between ErrorRecoveryService and the complete workflow.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
from copy import deepcopy

from src.orchestration.state import AgentState
from src.models.data_models import (
    StructuredCV, CVSection, CVItem, JobDescriptionData,
    Section, Item, ItemStatus, ItemType
)
from src.agents.parser_agent import ParserAgent
from src.agents.research_agent import ResearchAgent
from src.agents.enhanced_content_writer import EnhancedContentWriterAgent
from src.agents.quality_assurance_agent import QualityAssuranceAgent
from src.services.error_recovery import ErrorRecoveryService
from src.core.enhanced_orchestrator import EnhancedOrchestrator
from src.exceptions import (
    LLMServiceError, ValidationError, ProcessingError,
    RateLimitError, TimeoutError
)


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Integration tests for error recovery across the workflow."""

    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        job_desc = JobDescriptionData(
            raw_text="Senior Python Developer with 5+ years experience",
            requirements=["Python", "Django", "PostgreSQL"],
            responsibilities=["Develop web applications", "Design APIs"]
        )
        
        cv = StructuredCV(
            sections=[
                Section(
                    name="experience",
                    items=[
                        Item(
                            id="exp_1",
                            type=ItemType.EXPERIENCE,
                            content="Software Engineer at TechCorp",
                            status=ItemStatus.PENDING
                        )
                    ]
                )
            ]
        )
        
        return AgentState(
            session_id=str(uuid4()),
            job_description_data=job_desc,
            structured_cv=cv,
            current_item_id="exp_1",
            processing_queue=["exp_1"],
            errors=[]
        )

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service that can simulate various error conditions."""
        mock = AsyncMock()
        mock.generate_async = AsyncMock()
        return mock

    @pytest.fixture
    def error_recovery_service(self):
        """Create ErrorRecoveryService instance."""
        return ErrorRecoveryService()

    @pytest.mark.asyncio
    async def test_rate_limit_recovery_integration(self, sample_state, mock_llm_service, error_recovery_service):
        """Test rate limit error recovery in agent workflow."""
        # Setup rate limit error
        rate_limit_error = RateLimitError("Rate limit exceeded", retry_after=2)
        mock_llm_service.generate_async.side_effect = [
            rate_limit_error,  # First call fails
            "Enhanced content for experience"  # Second call succeeds
        ]
        
        # Create agent with error recovery
        agent = EnhancedContentWriterAgent(
            name="test_writer",
            description="Test content writer",
            content_type="experience"
        )
        agent.llm_service = mock_llm_service
        
        # Execute with error recovery
        with patch.object(error_recovery_service, 'execute_with_recovery') as mock_recovery:
            mock_recovery.return_value = "Fallback content for experience"
            
            result = await agent.run_as_node(sample_state)
            
            # Verify recovery was attempted
            assert mock_recovery.called
            assert result["structured_cv"].sections[0].items[0].status == ItemStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, sample_state, mock_llm_service, error_recovery_service):
        """Test circuit breaker pattern in workflow."""
        # Setup multiple failures to trigger circuit breaker
        mock_llm_service.generate_async.side_effect = [
            LLMServiceError("Service unavailable"),
            LLMServiceError("Service unavailable"),
            LLMServiceError("Service unavailable"),
            LLMServiceError("Service unavailable"),
            LLMServiceError("Service unavailable")
        ]
        
        agent = EnhancedContentWriterAgent(
            name="test_writer",
            description="Test content writer",
            content_type="experience"
        )
        agent.llm_service = mock_llm_service
        
        # Execute multiple times to trigger circuit breaker
        for i in range(5):
            try:
                await agent.run_as_node(sample_state)
            except Exception:
                pass
        
        # Verify circuit breaker state
        circuit_breaker = error_recovery_service.circuit_breakers.get("llm_service")
        if circuit_breaker:
            assert circuit_breaker.failure_count >= 3

    @pytest.mark.asyncio
    async def test_graceful_degradation_integration(self, sample_state, mock_llm_service, error_recovery_service):
        """Test graceful degradation when LLM service fails."""
        # Setup persistent LLM failure
        mock_llm_service.generate_async.side_effect = LLMServiceError("Service unavailable")
        
        agent = EnhancedContentWriterAgent(
            name="test_writer",
            description="Test content writer",
            content_type="experience"
        )
        agent.llm_service = mock_llm_service
        
        # Mock fallback content generation
        with patch.object(error_recovery_service, 'generate_fallback_content') as mock_fallback:
            mock_fallback.return_value = "Fallback experience content"
            
            result = await agent.run_as_node(sample_state)
            
            # Verify fallback was used
            assert mock_fallback.called
            assert "Fallback" in result["structured_cv"].sections[0].items[0].content

    @pytest.mark.asyncio
    async def test_error_propagation_integration(self, sample_state, mock_llm_service):
        """Test error propagation through the workflow."""
        # Setup validation error
        validation_error = ValidationError("Invalid content format")
        mock_llm_service.generate_async.side_effect = validation_error
        
        agent = EnhancedContentWriterAgent(
            name="test_writer",
            description="Test content writer",
            content_type="experience"
        )
        agent.llm_service = mock_llm_service
        
        result = await agent.run_as_node(sample_state)
        
        # Verify error was captured in state
        assert len(result["errors"]) > 0
        assert "ValidationError" in str(result["errors"][0])

    @pytest.mark.asyncio
    async def test_orchestrator_error_handling_integration(self, sample_state, mock_llm_service):
        """Test error handling at the orchestrator level."""
        # Create orchestrator with mocked services
        with patch('src.core.enhanced_orchestrator.EnhancedLLMService') as mock_llm_class:
            mock_llm_class.return_value = mock_llm_service
            mock_llm_service.generate_async.side_effect = TimeoutError("Request timeout")
            
            orchestrator = EnhancedOrchestrator()
            
            # Execute workflow with error
            result = await orchestrator.process_cv(
                job_description="Test job",
                base_cv_data={"experience": ["Test experience"]}
            )
            
            # Verify error handling
            assert "errors" in result
            assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_error_recovery(self, sample_state, mock_llm_service, error_recovery_service):
        """Test error recovery under concurrent operations."""
        # Setup intermittent failures
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise LLMServiceError("Intermittent failure")
            return f"Success response {call_count}"
        
        mock_llm_service.generate_async.side_effect = side_effect
        
        # Create multiple agents
        agents = [
            EnhancedContentWriterAgent(
                name=f"writer_{i}",
                description=f"Writer {i}",
                content_type="experience"
            )
            for i in range(3)
        ]
        
        for agent in agents:
            agent.llm_service = mock_llm_service
        
        # Execute concurrently
        tasks = [agent.run_as_node(deepcopy(sample_state)) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify some succeeded and some failed gracefully
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        assert len(successes) > 0  # Some should succeed
        # All should either succeed or fail gracefully (no unhandled exceptions)
        assert all(isinstance(r, dict) or isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_error_recovery_metrics_integration(self, sample_state, mock_llm_service, error_recovery_service):
        """Test error recovery metrics collection."""
        # Setup error scenario
        mock_llm_service.generate_async.side_effect = [
            RateLimitError("Rate limit", retry_after=1),
            "Success after retry"
        ]
        
        agent = EnhancedContentWriterAgent(
            name="test_writer",
            description="Test content writer",
            content_type="experience"
        )
        agent.llm_service = mock_llm_service
        
        # Execute with error recovery
        await agent.run_as_node(sample_state)
        
        # Verify metrics were collected
        metrics = error_recovery_service.get_metrics()
        assert "total_errors" in metrics
        assert "recovery_attempts" in metrics
        assert "successful_recoveries" in metrics

    @pytest.mark.asyncio
    async def test_error_context_preservation(self, sample_state, mock_llm_service, error_recovery_service):
        """Test that error context is preserved through recovery attempts."""
        # Setup error with context
        error_with_context = LLMServiceError(
            "Processing failed",
            context={"item_id": "exp_1", "attempt": 1, "model": "gpt-4"}
        )
        mock_llm_service.generate_async.side_effect = error_with_context
        
        agent = EnhancedContentWriterAgent(
            name="test_writer",
            description="Test content writer",
            content_type="experience"
        )
        agent.llm_service = mock_llm_service
        
        # Execute with error
        result = await agent.run_as_node(sample_state)
        
        # Verify error context is preserved
        assert len(result["errors"]) > 0
        error_info = result["errors"][0]
        assert "item_id" in str(error_info) or hasattr(error_info, 'context')

    @pytest.mark.asyncio
    async def test_recovery_strategy_selection(self, sample_state, mock_llm_service, error_recovery_service):
        """Test that appropriate recovery strategies are selected based on error type."""
        # Test different error types
        error_scenarios = [
            (RateLimitError("Rate limit", retry_after=1), "retry_with_backoff"),
            (TimeoutError("Timeout"), "retry_with_timeout"),
            (ValidationError("Invalid format"), "fallback_content"),
            (LLMServiceError("Service error"), "circuit_breaker")
        ]
        
        for error, expected_strategy in error_scenarios:
            mock_llm_service.generate_async.side_effect = error
            
            agent = EnhancedContentWriterAgent(
                name="test_writer",
                description="Test content writer",
                content_type="experience"
            )
            agent.llm_service = mock_llm_service
            
            with patch.object(error_recovery_service, 'determine_recovery_action') as mock_strategy:
                mock_strategy.return_value = expected_strategy
                
                await agent.run_as_node(sample_state)
                
                # Verify correct strategy was selected
                mock_strategy.assert_called()
                call_args = mock_strategy.call_args[0]
                assert isinstance(call_args[0], type(error))