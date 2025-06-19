"""Tests for simplified ItemProcessor.

Tests the ItemProcessor after retry consolidation - it should now
trust the LLM service to handle all resilience and not implement
its own retry logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.services.item_processor import ItemProcessor
from src.services.llm_service import LLMResponse
from src.models.data_models import Item, ProcessingStatus, ItemStatus
from src.utils.exceptions import ConfigurationError


class TestItemProcessorSimplified:
    """Test simplified ItemProcessor without retry logic."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        mock_service = Mock()
        mock_service.generate_content = AsyncMock()
        return mock_service

    @pytest.fixture
    def item_processor(self, mock_llm_service):
        """Create ItemProcessor instance for testing."""
        return ItemProcessor(llm_client=mock_llm_service)

    @pytest.fixture
    def sample_item(self):
        """Create a sample item for testing."""
        # ItemMetadata removed - using Item.metadata dict instead
        
        return Item(
            content="Sample qualification content",
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": "test-item-1",
                "item_type": "qualification",
                "title": "Software Engineer",
                "description": "Test job"
            }
        )

    @pytest.fixture
    def job_context(self):
        """Create job context for testing."""
        return {
            "job_title": "Software Engineer",
            "company": "Test Company",
            "requirements": ["Python", "Django", "REST APIs"]
        }

    @pytest.mark.asyncio
    async def test_successful_item_processing(self, item_processor, sample_item, job_context, mock_llm_service):
        """Test successful item processing without retries."""
        # Mock successful LLM response
        mock_response = LLMResponse(
            success=True,
            content="Enhanced qualification content",
            metadata={"processing_time": 1.5}
        )
        mock_llm_service.generate_content.return_value = mock_response
        
        result = await item_processor.process_item(sample_item, job_context)
        
        assert result is True
        assert sample_item.metadata.status == ProcessingStatus.COMPLETED
        # Verify LLM service was called exactly once (no retries)
        assert mock_llm_service.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_failed_item_processing_no_retries(self, item_processor, sample_item, job_context, mock_llm_service):
        """Test that ItemProcessor doesn't retry on LLM service failures."""
        # Mock failed LLM response (LLM service already handled retries)
        mock_response = LLMResponse(
            success=False,
            content=None,
            error_message="LLM service failed after retries"
        )
        mock_llm_service.generate_content.return_value = mock_response
        
        result = await item_processor.process_item(sample_item, job_context)
        
        assert result is False
        assert sample_item.metadata.status == ProcessingStatus.FAILED
        # Verify LLM service was called exactly once (no additional retries)
        assert mock_llm_service.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_item_not_pending_status(self, item_processor, sample_item, job_context, mock_llm_service):
        """Test handling of items not in pending status."""
        # Set item to already processed
        sample_item.metadata.update_status(ProcessingStatus.COMPLETED)
        
        result = await item_processor.process_item(sample_item, job_context)
        
        assert result is False
        # LLM service should not be called
        assert mock_llm_service.generate_content.call_count == 0

    @pytest.mark.asyncio
    async def test_configuration_error_propagation(self, item_processor, sample_item, job_context, mock_llm_service):
        """Test that configuration errors are properly propagated."""
        # Mock LLM service to raise configuration error
        mock_llm_service.generate_content.side_effect = ConfigurationError("Invalid API key")
        
        with pytest.raises(ConfigurationError, match="Invalid API key"):
            await item_processor.process_item(sample_item, job_context)
        
        # Verify LLM service was called exactly once
        assert mock_llm_service.generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_qualification_content_generation(self, item_processor, mock_llm_service):
        """Test qualification content generation."""
        # Create qualification item
        qualification_item = Item(
            content="Python programming",
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": "qual-1",
                "item_type": "qualification",
                "skill": "Python",
                "level": "Expert"
            }
        )
        
        job_context = {
            "job_title": "Python Developer",
            "requirements": ["Python", "Django"]
        }
        
        # Mock successful response
        mock_response = LLMResponse(
            success=True,
            content="Enhanced Python qualification",
            metadata={}
        )
        mock_llm_service.generate_content.return_value = mock_response
        
        result = await item_processor.process_item(qualification_item, job_context)
        
        assert result is True
        assert qualification_item.metadata.status == ProcessingStatus.COMPLETED
        
        # Verify the prompt was created correctly
        call_args = mock_llm_service.generate_content.call_args
        assert "prompt" in call_args.kwargs
        assert "Python" in call_args.kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_experience_content_generation(self, item_processor, mock_llm_service):
        """Test experience content generation."""
        # Create experience item
        experience_item = Item(
            content="Software Developer at TechCorp",
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": "exp-1",
                "item_type": "experience",
                "company": "TechCorp",
                "position": "Software Developer",
                "duration": "2 years"
            }
        )
        
        job_context = {
            "job_title": "Senior Developer",
            "company": "NewCorp"
        }
        
        # Mock successful response
        mock_response = LLMResponse(
            success=True,
            content="Enhanced experience description",
            metadata={}
        )
        mock_llm_service.generate_content.return_value = mock_response
        
        result = await item_processor.process_item(experience_item, job_context)
        
        assert result is True
        assert experience_item.metadata.status == ProcessingStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_project_content_generation(self, item_processor, mock_llm_service):
        """Test project content generation."""
        # Create project item
        project_item = Item(
            content="E-commerce platform",
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": "proj-1",
                "item_type": "project",
                "name": "E-commerce Platform",
                "technologies": ["Python", "Django", "PostgreSQL"]
            }
        )
        
        job_context = {
            "job_title": "Full Stack Developer",
            "requirements": ["Python", "Django", "Database"]
        }
        
        # Mock successful response
        mock_response = LLMResponse(
            success=True,
            content="Enhanced project description",
            metadata={}
        )
        mock_llm_service.generate_content.return_value = mock_response
        
        result = await item_processor.process_item(project_item, job_context)
        
        assert result is True
        assert project_item.metadata.status == ProcessingStatus.COMPLETED

    def test_item_processor_initialization(self):
        """Test ItemProcessor initialization without rate limiter."""
        mock_llm_service = Mock()
        processor = ItemProcessor(llm_client=mock_llm_service)
        
        assert processor.llm_client == mock_llm_service
        assert hasattr(processor, 'logger')
        assert hasattr(processor, 'settings')
        # Verify rate limiter is not initialized
        assert not hasattr(processor, 'rate_limiter')

    def test_qa_callback_integration(self):
        """Test QA callback integration."""
        mock_llm_service = Mock()
        mock_qa_callback = Mock()
        
        processor = ItemProcessor(
            llm_client=mock_llm_service,
            qa_callback=mock_qa_callback
        )
        
        assert processor.qa_callback == mock_qa_callback


class TestItemProcessorMetrics:
    """Test ItemProcessor metrics and statistics."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        mock_service = Mock()
        mock_service.generate_content = AsyncMock()
        return mock_service

    @pytest.fixture
    def item_processor(self, mock_llm_service):
        """Create ItemProcessor instance for testing."""
        return ItemProcessor(llm_client=mock_llm_service)

    @pytest.mark.asyncio
    async def test_processing_statistics_success(self, item_processor, mock_llm_service):
        """Test that processing statistics are updated on success."""
        # Create test item
        item = Item(
            content="Test content",
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": "test-1",
                "item_type": "qualification"
            }
        )
        
        # Mock successful response
        mock_response = LLMResponse(success=True, content="Enhanced content")
        mock_llm_service.generate_content.return_value = mock_response
        
        initial_processed = item_processor.total_processed
        
        result = await item_processor.process_item(item, {})
        
        assert result is True
        assert item_processor.total_processed == initial_processed + 1

    @pytest.mark.asyncio
    async def test_processing_statistics_failure(self, item_processor, mock_llm_service):
        """Test that processing statistics are updated on failure."""
        # Create test item
        item = Item(
            content="Test content",
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": "test-1",
                "item_type": "qualification"
            }
        )
        
        # Mock failed response
        mock_response = LLMResponse(success=False, error_message="Test error")
        mock_llm_service.generate_content.return_value = mock_response
        
        initial_failed = item_processor.total_failed
        
        result = await item_processor.process_item(item, {})
        
        assert result is False
        assert item_processor.total_failed == initial_failed + 1