"""Security utilities for credential redaction and sensitive data protection.

This module provides utilities to prevent sensitive information from being
exposed in logs, error messages, and other outputs.
"""

import re
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class RedactionConfig:
    """Configuration for data redaction."""

    # Sensitive field patterns (case-insensitive)
    sensitive_fields: List[str] = field(default_factory=lambda: [
        'api_key', 'apikey', 'api-key',
        'password', 'passwd', 'pwd',
        'secret', 'token', 'auth',
        'credential', 'key', 'private',
        'gemini_api_key', 'openai_api_key',
        'authorization', 'bearer',
        'session_id', 'session-id',
        'user_id', 'user-id',
        'email', 'phone', 'ssn',
        'credit_card', 'card_number'
    ])

    # Regex patterns for sensitive data
    sensitive_patterns: List[str] = field(default_factory=lambda: [
        r'AIza[0-9A-Za-z\-_]{35}',  # Google API keys
        r'sk-[a-zA-Z0-9]{48}',      # OpenAI API keys
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\b\d{3}-\d{2}-\d{4}\b',   # SSN format
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card format
        r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',  # UUIDs
    ])

    # Replacement text
    redaction_text: str = "[REDACTED]"

    # Whether to redact email addresses
    redact_emails: bool = False

    # Whether to redact UUIDs (session IDs, etc.)
    redact_uuids: bool = True


class CredentialRedactor:
    """Utility class for redacting sensitive information from data."""

    def __init__(self, config: Optional[RedactionConfig] = None):
        self.config = config or RedactionConfig()
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.sensitive_patterns
        ]

    def redact_string(self, text: str) -> str:
        """Redact sensitive information from a string.

        Args:
            text: String to redact

        Returns:
            String with sensitive information redacted
        """
        if not isinstance(text, str):
            return text

        redacted_text = text

        # Apply regex patterns
        for pattern in self._compiled_patterns:
            if pattern.pattern.startswith(r'\b[A-Za-z0-9._%+-]+@') and not self.config.redact_emails:
                continue
            if pattern.pattern.startswith(r'\b[0-9a-f]{8}-') and not self.config.redact_uuids:
                continue
            redacted_text = pattern.sub(self.config.redaction_text, redacted_text)

        return redacted_text

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from a dictionary.

        Args:
            data: Dictionary to redact

        Returns:
            Dictionary with sensitive information redacted
        """
        if not isinstance(data, dict):
            return data

        redacted_data = {}

        for key, value in data.items():
            # Check if key is sensitive
            if self._is_sensitive_field(key):
                redacted_data[key] = self.config.redaction_text
            elif isinstance(value, dict):
                redacted_data[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted_data[key] = self.redact_list(value)
            elif isinstance(value, str):
                redacted_data[key] = self.redact_string(value)
            else:
                redacted_data[key] = value

        return redacted_data

    def redact_list(self, data: List[Any]) -> List[Any]:
        """Redact sensitive information from a list.

        Args:
            data: List to redact

        Returns:
            List with sensitive information redacted
        """
        if not isinstance(data, list):
            return data

        redacted_list = []

        for item in data:
            if isinstance(item, dict):
                redacted_list.append(self.redact_dict(item))
            elif isinstance(item, list):
                redacted_list.append(self.redact_list(item))
            elif isinstance(item, str):
                redacted_list.append(self.redact_string(item))
            else:
                redacted_list.append(item)

        return redacted_list

    def redact_any(self, data: Any) -> Any:
        """Redact sensitive information from any data type.

        Args:
            data: Data to redact

        Returns:
            Data with sensitive information redacted
        """
        if isinstance(data, dict):
            return self.redact_dict(data)
        elif isinstance(data, list):
            return self.redact_list(data)
        elif isinstance(data, str):
            return self.redact_string(data)
        else:
            return data

    def redact_json(self, json_str: str) -> str:
        """Redact sensitive information from a JSON string.

        Args:
            json_str: JSON string to redact

        Returns:
            JSON string with sensitive information redacted
        """
        try:
            data = json.loads(json_str)
            redacted_data = self.redact_any(data)
            return json.dumps(redacted_data, indent=2)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, treat as regular string
            return self.redact_string(json_str)

    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field name indicates sensitive data.

        Args:
            field_name: Field name to check

        Returns:
            True if field is considered sensitive
        """
        field_lower = field_name.lower()
        return any(
            sensitive_field in field_lower
            for sensitive_field in self.config.sensitive_fields
        )


# Global redactor instance
_global_redactor = CredentialRedactor()


def redact_sensitive_data(data: Any, config: Optional[RedactionConfig] = None) -> Any:
    """Redact sensitive information from any data structure.

    Args:
        data: Data to redact
        config: Optional redaction configuration

    Returns:
        Data with sensitive information redacted
    """
    if config:
        redactor = CredentialRedactor(config)
    else:
        redactor = _global_redactor

    return redactor.redact_any(data)


def redact_log_message(message: str, **kwargs) -> str:
    """Redact sensitive information from a log message and its kwargs.

    Args:
        message: Log message
        **kwargs: Additional log data

    Returns:
        Redacted log message
    """
    # Redact the main message
    redacted_message = _global_redactor.redact_string(message)

    # Redact kwargs if present
    if kwargs:
        redacted_kwargs = _global_redactor.redact_dict(kwargs)
        redacted_message += f" | {json.dumps(redacted_kwargs)}"

    return redacted_message


def safe_str(obj: Any) -> str:
    """Convert object to string with sensitive data redacted.

    Args:
        obj: Object to convert to string

    Returns:
        String representation with sensitive data redacted
    """
    try:
        if isinstance(obj, (dict, list)):
            redacted_obj = _global_redactor.redact_any(obj)
            return json.dumps(redacted_obj, indent=2, default=str)
        else:
            return _global_redactor.redact_string(str(obj))
    except Exception:
        return "[OBJECT_CONVERSION_ERROR]"


def create_safe_config_dict(config_obj: Any) -> Dict[str, Any]:
    """Create a safe dictionary representation of a configuration object.

    Args:
        config_obj: Configuration object

    Returns:
        Dictionary with sensitive fields redacted
    """
    try:
        if hasattr(config_obj, '__dict__'):
            config_dict = config_obj.__dict__.copy()
        elif hasattr(config_obj, '_asdict'):
            config_dict = config_obj._asdict()
        elif isinstance(config_obj, dict):
            config_dict = config_obj.copy()
        else:
            config_dict = {"config": str(config_obj)}

        return _global_redactor.redact_dict(config_dict)
    except Exception:
        return {"error": "Failed to create safe config representation"}


def mask_api_key(api_key: str, visible_chars: int = 4) -> str:
    """Mask an API key showing only the first few characters.

    Args:
        api_key: API key to mask
        visible_chars: Number of characters to show at the beginning

    Returns:
        Masked API key
    """
    if not api_key or len(api_key) <= visible_chars:
        return "[REDACTED]"

    return f"{api_key[:visible_chars]}{'*' * (len(api_key) - visible_chars)}"


def validate_no_secrets_in_logs(log_content: str) -> List[str]:
    """Validate that log content doesn't contain secrets.

    Args:
        log_content: Log content to validate

    Returns:
        List of potential security issues found
    """
    issues = []

    # Check for API key patterns
    api_key_patterns = [
        (r'AIza[0-9A-Za-z\-_]{35}', 'Google API key'),
        (r'sk-[a-zA-Z0-9]{48}', 'OpenAI API key'),
        (r'[a-zA-Z0-9]{32,}', 'Potential API key or token'),
    ]

    for pattern, description in api_key_patterns:
        if re.search(pattern, log_content):
            issues.append(f"Potential {description} found in logs")

    # Check for sensitive field names with values
    sensitive_patterns = [
        r'(?i)(api_key|password|secret|token)\s*[:=]\s*["\']?[a-zA-Z0-9]+',
        r'(?i)(authorization|bearer)\s*[:=]\s*["\']?[a-zA-Z0-9]+',
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, log_content):
            issues.append("Sensitive field with value found in logs")

    return issues
