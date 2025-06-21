"""Decorator utilities for the aicvgen project."""

import asyncio
from typing import Callable
import functools


def create_async_sync_decorator(
    async_wrapper_func: Callable, sync_wrapper_func: Callable
):
    """Helper to create decorators that work with both async and sync functions.
    This version supports decorating both methods and standalone functions.
    Handles correct binding for methods (self/cls) and functions.
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def method_async_wrapper(*args, **kwargs):
                bound_func = func
                return await async_wrapper_func(bound_func)(*args, **kwargs)

            return method_async_wrapper
        else:

            @functools.wraps(func)
            def method_sync_wrapper(*args, **kwargs):
                bound_func = func
                return sync_wrapper_func(bound_func)(*args, **kwargs)

            return method_sync_wrapper

    return decorator
