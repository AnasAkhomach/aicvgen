"""Decorator utilities for the aicvgen project."""

import asyncio
from typing import Callable


def create_async_sync_decorator(async_wrapper_func: Callable, sync_wrapper_func: Callable):
    """Helper to create decorators that work with both async and sync functions.
    
    This eliminates the common pattern of:
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
    
    Args:
        async_wrapper_func: Function that wraps async functions
        sync_wrapper_func: Function that wraps sync functions
        
    Returns:
        A decorator function
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            return async_wrapper_func(func)
        else:
            return sync_wrapper_func(func)
    return decorator