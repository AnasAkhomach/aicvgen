import logging
from functools import wraps
from typing import Callable, Type, Tuple

logger = logging.getLogger("error_utils")


def handle_errors(
    default_return=None, exceptions=(Exception,), log_level=logging.ERROR
) -> Callable:
    """Decorator to handle errors in a function, log, and optionally return a default value."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.log(log_level, f"Error in {func.__name__}: {e}")
                return default_return

        return wrapper

    return decorator


def try_or_log(func: Callable, *args, **kwargs) -> Tuple[bool, any]:
    """Try to run a function, log on error, and return (success, result)."""
    try:
        return True, func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}")
        return False, None
