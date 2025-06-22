"""Integration tests for ItemProcessor with real LLM service.

Tests the integration between ItemProcessor and EnhancedLLMService
after retry consolidation implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.services.item_processor import ItemProcessor
from src.services.llm_service import EnhancedLLMService, LLMResponse
from src.models.data_models import Item, ProcessingStatus, ItemStatus, MetadataModel
from src.utils.exceptions import ConfigurationError, RateLimitError


class TestItemProcessorLLMServiceIntegration:
    """Test ItemProcessor integration with real LLM service."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a real LLM service instance with mocked API calls."""
        with patch("src.services.llm_service.genai") as mock_genai:
            # Mock the genai module
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_genai.configure = Mock()
            # Provide a mock settings object
            mock_settings = Mock()
            mock_settings.llm.gemini_api_key_fallback = "fallback-key"
            mock_settings.llm_settings.default_model = "test-model"
            service = EnhancedLLMService(
                settings=mock_settings, user_api_key="test-api-key"
            )
            service.llm = mock_model

            return service

    @pytest.fixture
    def item_processor(self, mock_llm_service):
        """Create ItemProcessor with real LLM service."""
        return ItemProcessor(llm_client=mock_llm_service)

    @pytest.fixture
    def sample_qualification_item(self):
        """Create a sample qualification item."""
        return Item(
            content="Python programming and web development",
            status=ItemStatus.INITIAL,
            metadata=MetadataModel(
                company=None,
                position=None,
                location=None,
                start_date=None,
                end_date=None,
                extra={
                    "item_id": "qual-integration-1",
                    "item_type": "qualification",
                    "skill": "Python",
                    "level": "Expert",
                    "years_experience": 5,
                    "status": ProcessingStatus.PENDING,  # Add status to metadata
                },
                # Add status field directly if needed by your code
                status=ProcessingStatus.PENDING,
            ),
        )

    @pytest.fixture
    def sample_experience_item(self):
        """Create a sample experience item."""
        return Item(
            content="Senior Software Engineer at TechCorp",
            status=ItemStatus.INITIAL,
            metadata=MetadataModel(
                company="TechCorp",
                position="Senior Software Engineer",
                location=None,
                start_date=None,
                end_date=None,
                extra={
                    "item_id": "exp-integration-1",
                    "item_type": "experience",
                    "duration": "3 years",
                    "achievements": [
                        "Led team of 5 developers",
                        "Improved system performance by 40%",
                    ],
                    "status": ProcessingStatus.PENDING,  # Add status to metadata
                },
                status=ProcessingStatus.PENDING,
            ),
        )

    @pytest.fixture
    def job_context(self):
        """Create job context for testing."""
        return {
            "job_title": "Senior Python Developer",
            "company": "InnovateTech",
            "requirements": [
                "5+ years Python experience",
                "Django/Flask framework",
                "REST API development",
                "Team leadership",
            ],
            "preferred_skills": [
                "AWS/Cloud platforms",
                "Microservices architecture",
                "CI/CD pipelines",
            ],
        }

    @pytest.mark.asyncio
    async def test_successful_qualification_processing(
        self, item_processor, sample_qualification_item, job_context, mock_llm_service
    ):
        """Test successful qualification processing with real service integration."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.text = (
            "Enhanced Python qualification tailored for Senior Python Developer role"
        )
        mock_llm_service.llm.generate_content_async = Mock(return_value=mock_response)

        result = await item_processor.process_item(
            sample_qualification_item, job_context
        )
        assert result is True
        assert sample_qualification_item.status == ProcessingStatus.COMPLETED

        # Verify LLM service was called with appropriate prompt
        mock_llm_service.llm.generate_content_async.assert_called_once()
        call_args = mock_llm_service.llm.generate_content_async.call_args[1]
        assert "Python" in str(call_args)
        assert "Senior Python Developer" in str(call_args)

    @pytest.mark.asyncio
    async def test_successful_experience_processing(
        self, item_processor, sample_experience_item, job_context, mock_llm_service
    ):
        """Test successful experience processing with real service integration."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.text = "Enhanced experience description highlighting leadership and Python expertise"
        mock_llm_service.llm.generate_content_async = Mock(return_value=mock_response)

        result = await item_processor.process_item(sample_experience_item, job_context)
        assert result is True
        assert sample_experience_item.status == ProcessingStatus.COMPLETED

        # Verify LLM service was called
        mock_llm_service.llm.generate_content_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_service_failure_handling(
        self, item_processor, sample_qualification_item, job_context, mock_llm_service
    ):
        """Test that ItemProcessor properly handles LLM service failures."""
        # Mock LLM service to return failed response (after its own retries)
        with patch.object(mock_llm_service, "generate_content") as mock_generate:
            mock_generate.return_value = LLMResponse(
                success=False,
                content="",
                error_message="LLM service failed after retries",
            )
            result = await item_processor.process_item(
                sample_qualification_item, job_context
            )
            assert result is False
            assert sample_qualification_item.status == ProcessingStatus.FAILED

            # Verify ItemProcessor called LLM service exactly once (no additional retries)
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_configuration_error_propagation(
        self, item_processor, sample_qualification_item, job_context
    ):
        """Test that configuration errors are properly propagated from LLM service."""
        # Create LLM service that will fail on initialization
        mock_settings = Mock()
        mock_settings.llm.gemini_api_key_fallback = "fallback-key"
        mock_settings.llm_settings.default_model = "test-model"
        with pytest.raises(ConfigurationError):
            service = EnhancedLLMService(settings=mock_settings, user_api_key=None)
            # Explicitly call a method that triggers the error if not raised in __init__
            service.validate_api_key()

    @pytest.mark.asyncio
    async def test_multiple_items_processing(
        self, item_processor, job_context, mock_llm_service
    ):
        """Test processing multiple items in sequence."""
        # Create multiple items
        items = []
        for i in range(3):
            items.append(
                Item(
                    content=f"Skill {i}",
                    status=ItemStatus.INITIAL,
                    metadata={
                        "item_id": f"item-{i}",
                        "item_type": "qualification",
                        "skill": f"Skill{i}",
                    },
                )
            )

        # Mock successful responses
        mock_response = Mock()
        mock_response.text = "Enhanced content"
        mock_llm_service.llm.generate_content_async = Mock(return_value=mock_response)

        # Process all items
        results = []
        for item in items:
            result = await item_processor.process_item(item, job_context)
            results.append(result)

        # Verify all items were processed successfully
        assert all(results)
        assert all(item.status == ProcessingStatus.COMPLETED for item in items)

        # Verify LLM service was called for each item
        assert mock_llm_service.llm.generate_content_async.call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_item_processing(
        self, item_processor, job_context, mock_llm_service
    ):
        """Test concurrent processing of multiple items."""
        # Create multiple items
        items = []
        for i in range(3):
            items.append(
                Item(
                    content=f"Concurrent skill {i}",
                    status=ItemStatus.INITIAL,
                    metadata={
                        "item_id": f"concurrent-item-{i}",
                        "item_type": "qualification",
                        "skill": f"ConcurrentSkill{i}",
                    },
                )
            )

        # Mock successful responses with delay to simulate real processing
        async def mock_generate_content(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return LLMResponse(
                success=True, content="Enhanced concurrent content", metadata={}
            )

        mock_llm_service.generate_content = mock_generate_content

        # Process items concurrently
        tasks = [item_processor.process_item(item, job_context) for item in items]

        results = await asyncio.gather(*tasks)

        # Verify all items were processed successfully
        assert all(results)
        assert all(item.status == ProcessingStatus.COMPLETED for item in items)

    @pytest.mark.asyncio
    async def test_processing_statistics_integration(
        self, item_processor, sample_qualification_item, job_context, mock_llm_service
    ):
        """Test that processing statistics are properly maintained during integration."""
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Enhanced content"
        mock_llm_service.llm.generate_content_async = Mock(return_value=mock_response)

        initial_processed = item_processor.total_processed
        initial_failed = item_processor.total_failed

        # Process item successfully
        result = await item_processor.process_item(
            sample_qualification_item, job_context
        )
        assert result is True
        assert item_processor.total_processed == initial_processed + 1
        assert item_processor.total_failed == initial_failed

        # Now test failure case
        sample_qualification_item.status = ItemStatus.INITIAL  # Reset for retry
        with patch.object(mock_llm_service, "generate_content") as mock_generate:
            mock_generate.return_value = LLMResponse(
                success=False, content="", error_message="Test failure"
            )
            result = await item_processor.process_item(
                sample_qualification_item, job_context
            )
            assert result is False
            assert item_processor.total_failed == initial_failed + 1


class TestItemProcessorErrorRecovery:
    """Test error recovery patterns in ItemProcessor integration."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a real LLM service instance with mocked API calls."""
        with patch("src.services.llm_service.genai") as mock_genai:
            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_genai.configure = Mock()
            mock_settings = Mock()
            mock_settings.llm.gemini_api_key_fallback = "fallback-key"
            mock_settings.llm_settings.default_model = "test-model"
            service = EnhancedLLMService(
                settings=mock_settings, user_api_key="test-api-key"
            )
            service.llm = mock_model

            return service

    @pytest.fixture
    def item_processor(self, mock_llm_service):
        """Create ItemProcessor with real LLM service."""
        return ItemProcessor(llm_client=mock_llm_service)

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_service_failure(
        self, item_processor, mock_llm_service
    ):
        """Test graceful degradation when LLM service fails completely."""
        # Create test item
        item = Item(
            content="Test content",
            status=ItemStatus.INITIAL,
            metadata={"item_id": "degradation-test", "item_type": "qualification"},
            raw_data={},
        )

        # Mock complete service failure
        with patch.object(mock_llm_service, "generate_content") as mock_generate:
            mock_generate.return_value = LLMResponse(
                success=False, content="", error_message="Complete service failure"
            )
            result = await item_processor.process_item(item, {})
            assert result is False
            assert item.status == ProcessingStatus.FAILED
            # Should not retry (trust LLM service to handle retries)
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_handling_integration(self, item_processor, mock_llm_service):
        """Test timeout handling in integration scenario."""
        # Create test item
        item = Item(
            content="Test content",
            status=ItemStatus.INITIAL,
            metadata={"item_id": "timeout-test", "item_type": "qualification"},
        )

        # Mock timeout scenario (LLM service should handle this internally)
        with patch.object(mock_llm_service, "generate_content") as mock_generate:
            mock_generate.return_value = LLMResponse(
                success=False, content="", error_message="Request timeout after retries"
            )
            result = await item_processor.process_item(item, {})
            assert result is False
            assert item.status == ProcessingStatus.FAILED
