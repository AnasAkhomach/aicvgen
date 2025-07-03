import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.services.llm_caching_service import LLMCachingService
from src.models.data_models import ContentType, LLMResponse
from src.models.llm_service_models import LLMCacheEntry


class TestLLMCachingService:
    """Test cases for LLMCachingService."""

    @pytest.fixture
    def temp_cache_file(self):
        """Create a temporary cache file for testing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def caching_service(self, temp_cache_file):
        """Create a LLMCachingService instance for testing."""
        return LLMCachingService(
            max_size=10, default_ttl_hours=1, persist_file=temp_cache_file
        )

    @pytest.mark.asyncio
    async def test_initialization(self, caching_service):
        """Test service initialization."""
        assert caching_service.max_size == 10
        assert caching_service.default_ttl_hours == 1
        assert not caching_service._initialized

        await caching_service.initialize()
        assert caching_service._initialized

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, caching_service):
        """Test cache key generation is consistent."""
        key1 = caching_service._generate_cache_key(
            "test prompt", "model1", temperature=0.7
        )
        key2 = caching_service._generate_cache_key(
            "test prompt", "model1", temperature=0.7
        )
        key3 = caching_service._generate_cache_key(
            "different prompt", "model1", temperature=0.7
        )

        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, caching_service):
        """Test basic cache set and get operations."""
        await caching_service.initialize()

        prompt = "test prompt"
        model = "test-model"
        response_data = {"content": "test response", "tokens_used": 10}

        # Set cache entry
        await caching_service.set(prompt, model, response_data, ttl_hours=1)

        # Get cache entry
        cached_response = await caching_service.get(prompt, model)
        assert cached_response == response_data

    @pytest.mark.asyncio
    async def test_cache_miss(self, caching_service):
        """Test cache miss behavior."""
        await caching_service.initialize()

        cached_response = await caching_service.get("nonexistent prompt", "model")
        assert cached_response is None

    @pytest.mark.asyncio
    async def test_cache_expiry(self, caching_service):
        """Test cache entry expiration."""
        await caching_service.initialize()

        prompt = "test prompt"
        model = "test-model"
        response_data = {"content": "test response"}

        # Create an expired entry
        cache_key = caching_service._generate_cache_key(prompt, model)
        expired_entry = LLMCacheEntry(
            response=response_data,
            expiry=datetime.now() - timedelta(hours=1),  # Expired
            created_at=datetime.now() - timedelta(hours=2),
            access_count=1,
            cache_key=cache_key,
        )

        async with caching_service._lock:
            caching_service._cache[cache_key] = expired_entry

        # Try to get expired entry
        cached_response = await caching_service.get(prompt, model)
        assert cached_response is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self, caching_service):
        """Test LRU eviction when cache is full."""
        await caching_service.initialize()

        # Fill cache to max capacity
        for i in range(caching_service.max_size):
            await caching_service.set(
                f"prompt_{i}", "model", {"content": f"response_{i}"}
            )

        # Add one more entry to trigger eviction
        await caching_service.set("new_prompt", "model", {"content": "new_response"})

        # First entry should be evicted
        first_entry = await caching_service.get("prompt_0", "model")
        assert first_entry is None

        # New entry should be present
        new_entry = await caching_service.get("new_prompt", "model")
        assert new_entry is not None

    @pytest.mark.asyncio
    async def test_cache_stats(self, caching_service):
        """Test cache statistics tracking."""
        await caching_service.initialize()

        # Initial stats
        stats = await caching_service.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

        # Add cache entry and access it
        await caching_service.set("prompt", "model", {"content": "response"})
        await caching_service.get("prompt", "model")  # Hit
        await caching_service.get("nonexistent", "model")  # Miss

        stats = await caching_service.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_check_cache_method(self, caching_service):
        """Test the check_cache method that returns LLMResponse."""
        await caching_service.initialize()

        prompt = "test prompt"
        model = "test-model"
        content_type = ContentType.CV_ANALYSIS

        # Cache miss
        result = await caching_service.check_cache(
            prompt, model, content_type, session_id="123"
        )
        assert result is None

        # Set cache entry
        llm_response = LLMResponse(
            content="test response",
            tokens_used=10,
            processing_time=0.5,
            model_used=model,
            success=True,
            metadata={"test": "data"},
        )

        await caching_service.cache_response(prompt, model, content_type, llm_response, session_id="123")

        # Cache hit
        result = await caching_service.check_cache(
            prompt, model, content_type, session_id="123"
        )
        assert result is not None
        assert isinstance(result, LLMResponse)
        assert result.content == "test response"
        assert result.metadata["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_clear_cache(self, caching_service):
        """Test cache clearing functionality."""
        await caching_service.initialize()

        # Add some entries
        await caching_service.set("prompt1", "model", {"content": "response1"})
        await caching_service.set("prompt2", "model", {"content": "response2"})

        stats_before = await caching_service.get_cache_stats()
        assert stats_before["size"] == 2

        # Clear cache
        await caching_service.clear()

        stats_after = await caching_service.get_cache_stats()
        assert stats_after["size"] == 0
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0

    @pytest.mark.asyncio
    async def test_evict_expired_entries(self, caching_service):
        """Test manual expiry eviction."""
        await caching_service.initialize()

        # Add valid and expired entries
        await caching_service.set("valid_prompt", "model", {"content": "valid"})

        # Manually add expired entry
        cache_key = caching_service._generate_cache_key("expired_prompt", "model")
        expired_entry = LLMCacheEntry(
            response={"content": "expired"},
            expiry=datetime.now() - timedelta(hours=1),
            created_at=datetime.now() - timedelta(hours=2),
            access_count=1,
            cache_key=cache_key,
        )

        async with caching_service._lock:
            caching_service._cache[cache_key] = expired_entry

        stats_before = await caching_service.get_cache_stats()
        assert stats_before["size"] == 2

        # Evict expired entries
        await caching_service.evict_expired_entries()

        stats_after = await caching_service.get_cache_stats()
        assert stats_after["size"] == 1  # Only valid entry should remain

    @pytest.mark.asyncio
    async def test_persistence_mock(self, temp_cache_file):
        """Test cache persistence with mocked file operations."""
        caching_service = LLMCachingService(persist_file=temp_cache_file)
        await caching_service.initialize()

        # Add some data
        await caching_service.set("prompt", "model", {"content": "response"})

        # Force save
        await caching_service._save_cache()

        # Verify file exists
        assert os.path.exists(temp_cache_file)
