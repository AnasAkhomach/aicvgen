import os
import asyncio
import hashlib
import json
import pickle
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import OrderedDict

from src.config.logging_config import get_structured_logger
from src.models.llm_service_models import LLMCacheEntry
from src.models.workflow_models import ContentType
from src.models.llm_data_models import LLMResponse
from src.config.settings import get_config

logger = get_structured_logger("llm_caching_service")


class LLMCachingService:
    """
    Advanced caching system with LRU eviction, TTL, and persistence.
    Handles all LLM response caching operations.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_hours: int = 24,
        persist_file: Optional[str] = None,
    ):
        self.max_size = max_size
        self.default_ttl_hours = default_ttl_hours
        self.persist_file = persist_file
        self._cache: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._initialized = False

        logger.info(
            "LLM caching service created",
            max_size=max_size,
            default_ttl_hours=default_ttl_hours,
            persist_file=persist_file,
        )

    async def initialize(self):
        """Asynchronously load persisted cache from file."""
        if self.persist_file and not self._initialized:
            await self._load_cache()
            self._initialized = True
            logger.info("LLM caching service initialized")

    def _generate_cache_key(
        self, prompt: str, model: str, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate a comprehensive cache key."""
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            **kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _is_expired(self, entry: LLMCacheEntry) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > entry.expiry

    def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items() if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self._cache[key]
            self._evictions += 1

    def _evict_lru(self):
        """Evict least recently used entries if cache is full."""
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (LRU)
            self._evictions += 1

    async def get(self, prompt: str, model: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        cache_key = self._generate_cache_key(prompt, model, **kwargs)
        async with self._lock:
            self._evict_expired()
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if not self._is_expired(entry):
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    return entry.response
                else:
                    del self._cache[cache_key]
                    self._evictions += 1
            self._misses += 1
            return None

    async def set(
        self,
        prompt: str,
        model: str,
        response: Dict[str, Any],
        ttl_hours: Optional[int] = None,
        **kwargs,
    ):
        """Cache response with TTL."""
        cache_key = self._generate_cache_key(prompt, model, **kwargs)
        ttl = ttl_hours or self.default_ttl_hours
        async with self._lock:
            self._evict_expired()
            self._evict_lru()
            entry = LLMCacheEntry(
                response=response,
                expiry=datetime.now() + timedelta(hours=ttl),
                created_at=datetime.now(),
                access_count=1,
                cache_key=cache_key,
            )
            self._cache[cache_key] = entry
            if self.persist_file:
                await self._save_cache()

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": hit_rate,
                "evictions": self._evictions,
                "memory_usage_estimate_mb": self._estimate_memory_usage(),
            }

    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB."""
        try:
            return len(self._cache) * 0.1  # Assume ~100KB per entry
        except (TypeError, ValueError):
            return 0.0

    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

            if self.persist_file:
                await self._save_cache()

    async def evict_expired_entries(self):
        """Public method to manually trigger eviction of expired entries."""
        async with self._lock:
            self._evict_expired()

    async def check_cache(
        self,
        prompt: str,
        model: str,
        content_type: ContentType,
        **kwargs,
    ) -> Optional[LLMResponse]:
        """
        Check cache for existing response and return as LLMResponse.

        Args:
            prompt: The prompt text
            model: The model name
            content_type: Type of content being generated
            **kwargs: Additional parameters for cache key generation

        Returns:
            LLMResponse if cached, None otherwise
        """
        cached_response = await self.get(
            prompt, model, content_type=content_type.value, **kwargs
        )

        if cached_response:
            self._hits += 1
            session_id = kwargs.get("session_id")
            item_id = kwargs.get("item_id")
            trace_id = kwargs.get("trace_id")

            logger.info(
                "Cache hit for LLM request",
                trace_id=trace_id,
                session_id=session_id,
                item_id=item_id,
                content_type=content_type.value,
            )

            # Ensure metadata exists
            if "metadata" not in cached_response:
                cached_response["metadata"] = {}
                
            cached_response["metadata"]["cache_hit"] = True
            cached_response["metadata"]["session_id"] = session_id
            cached_response["metadata"]["item_id"] = item_id
            cached_response["processing_time"] = 0.001
            return LLMResponse(**cached_response)

        self._misses += 1
        logger.debug(
            "Cache miss for LLM request",
            trace_id=kwargs.get("trace_id"),
            session_id=kwargs.get("session_id"),
            item_id=kwargs.get("item_id"),
            content_type=content_type.value,
        )
        return None

    async def cache_response(
        self,
        prompt: str,
        model: str,
        content_type: ContentType,
        llm_response: LLMResponse,
        **kwargs,
    ) -> None:
        """Cache the successful LLM response."""
        cache_data = llm_response.model_dump()
        cache_data["metadata"]["content_type"] = content_type.value
        cache_data["metadata"]["cache_hit"] = False

        await self.set(
            prompt,
            model,
            cache_data,
            ttl_hours=2,
            content_type=content_type.value,
            **kwargs,
        )

    async def _save_cache(self):
        """Persist cache to file asynchronously."""
        if not self.persist_file:
            return
        try:
            cache_data = {
                "cache": dict(self._cache),
                "stats": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "evictions": self._evictions,
                },
                "saved_at": datetime.now().isoformat(),
            }

            def _dump():
                with open(self.persist_file, "wb") as f:
                    pickle.dump(cache_data, f)

            await asyncio.to_thread(_dump)
        except (IOError, pickle.PickleError) as e:
            logger.warning("Failed to save cache", error=str(e))

    async def _load_cache(self):
        """Load persisted cache from file asynchronously."""
        if not self.persist_file or not os.path.exists(self.persist_file):
            return

        try:
            # Check if file is empty
            if os.path.getsize(self.persist_file) == 0:
                logger.info("Cache file is empty, skipping load")
                return

            def _load():
                with open(self.persist_file, "rb") as f:
                    return pickle.load(f)

            cache_data = await asyncio.to_thread(_load)

            # Restore cache entries that haven't expired
            for key, entry in cache_data.get("cache", {}).items():
                if not self._is_expired(entry):
                    self._cache[key] = entry

            # Restore stats
            stats = cache_data.get("stats", {})
            self._hits = stats.get("hits", 0)
            self._misses = stats.get("misses", 0)
            self._evictions = stats.get("evictions", 0)

            logger.info(
                "Cache loaded from persistence",
                entries_loaded=len(self._cache),
                file=self.persist_file,
            )
        except (IOError, pickle.PickleError, EOFError) as e:
            logger.warning("Failed to load persisted cache", error=str(e))


# Global caching service instance
_cache_lock = threading.Lock()
_CACHING_SERVICE: Optional[LLMCachingService] = None


def get_llm_caching_service() -> LLMCachingService:
    """Get global LLM caching service instance."""
    global _CACHING_SERVICE  # pylint: disable=global-statement
    if _CACHING_SERVICE is None:
        with _cache_lock:
            if _CACHING_SERVICE is None:


                settings = get_config()
                cache_file = (
                    os.path.join(settings.data_dir, "llm_cache.pkl")
                    if hasattr(settings, "data_dir")
                    else None
                )
                _CACHING_SERVICE = LLMCachingService(persist_file=cache_file)
    return _CACHING_SERVICE
