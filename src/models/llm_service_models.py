from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class LLMCacheEntry(BaseModel):
    response: dict
    expiry: datetime
    created_at: datetime
    access_count: int
    cache_key: str


class LLMApiKeyInfo(BaseModel):
    using_user_key: bool
    using_fallback: bool
    has_fallback_available: bool
    key_source: str


class LLMServiceStats(BaseModel):
    total_calls: int
    total_tokens: int
    total_processing_time: float
    average_processing_time: float
    model_name: str
    rate_limiter_status: Optional[dict]
    cache_stats: dict
    optimizer_stats: dict
    async_stats: dict


class LLMPerformanceOptimizationResult(BaseModel):
    cache_optimization: dict
    timestamp: str
