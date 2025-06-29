"""Test for centralized CATCHABLE_EXCEPTIONS tuple refactoring."""

import pytest
from src.error_handling.exceptions import CATCHABLE_EXCEPTIONS, AicvgenError


def test_catchable_exceptions_centralized():
    """Test that CATCHABLE_EXCEPTIONS is properly centralized and accessible."""
    # Verify CATCHABLE_EXCEPTIONS is a tuple
    assert isinstance(CATCHABLE_EXCEPTIONS, tuple)

    # Verify it contains expected exception types
    assert ValueError in CATCHABLE_EXCEPTIONS
    assert TypeError in CATCHABLE_EXCEPTIONS
    assert KeyError in CATCHABLE_EXCEPTIONS
    assert AicvgenError in CATCHABLE_EXCEPTIONS

    # Verify it's non-empty
    assert len(CATCHABLE_EXCEPTIONS) > 0


def test_boundaries_imports_centralized_exceptions():
    """Test that boundaries.py correctly imports CATCHABLE_EXCEPTIONS from exceptions.py."""
    from src.error_handling.boundaries import (
        CATCHABLE_EXCEPTIONS as boundaries_exceptions,
    )
    from src.error_handling.exceptions import (
        CATCHABLE_EXCEPTIONS as exceptions_exceptions,
    )

    # Verify they're the same object
    assert boundaries_exceptions is exceptions_exceptions


def test_content_aggregator_imports_centralized_exceptions():
    """Test that content_aggregator.py correctly imports CATCHABLE_EXCEPTIONS from exceptions.py."""
    from src.core.content_aggregator import (
        CATCHABLE_EXCEPTIONS as aggregator_exceptions,
    )
    from src.error_handling.exceptions import (
        CATCHABLE_EXCEPTIONS as exceptions_exceptions,
    )

    # Verify they're the same object
    assert aggregator_exceptions is exceptions_exceptions


def test_no_duplicate_catchable_exceptions_definitions():
    """Test that there's only one definition of CATCHABLE_EXCEPTIONS in the codebase."""
    # This test ensures we've successfully eliminated duplication
    # The fact that the imports above work and reference the same object confirms this

    # Test that boundaries.py doesn't define its own CATCHABLE_EXCEPTIONS
    import src.error_handling.boundaries as boundaries_module
    import src.core.content_aggregator as aggregator_module
    import src.error_handling.exceptions as exceptions_module

    # All should reference the same object from exceptions module
    assert (
        boundaries_module.CATCHABLE_EXCEPTIONS is exceptions_module.CATCHABLE_EXCEPTIONS
    )
    assert (
        aggregator_module.CATCHABLE_EXCEPTIONS is exceptions_module.CATCHABLE_EXCEPTIONS
    )
