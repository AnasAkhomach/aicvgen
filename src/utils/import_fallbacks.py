"""Common import fallback utilities."""


def get_security_utils():
    """Get security utilities with fallback implementations.

    Returns:
        Tuple of (redact_sensitive_data, redact_log_message) functions
    """
    try:
        from .security_utils import redact_log_message, redact_sensitive_data

        return redact_sensitive_data, redact_log_message
    except ImportError:
        # Fallback implementations
        def redact_sensitive_data(data):
            return data

        def redact_log_message(message):
            return message

        return redact_sensitive_data, redact_log_message
