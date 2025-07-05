"""Decorator utilities for the aicvgen project."""

import asyncio
import functools
from typing import Callable


def create_async_sync_decorator(
    async_wrapper_func: Callable, sync_wrapper_func: Callable
):
    """Helper to create decorators that work with both async and sync functions.
    This version supports decorating both methods and standalone functions.
    Handles correct binding for methods (self/cls) and functions.
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            # Only wrap once, do not double-wrap
            return async_wrapper_func(func)
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                bound_func = func
                return sync_wrapper_func(bound_func)(*args, **kwargs)

            return sync_wrapper

    return decorator
