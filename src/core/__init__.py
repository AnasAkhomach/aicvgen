"""Core module for the aicvgen application."""

from .state_manager import StateManager
from .content_aggregator import ContentAggregator

__all__ = [
    "StateManager",
    "ContentAggregator"
]