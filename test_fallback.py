"""Test script for D-01 implementation"""

import asyncio
from src.services.error_recovery import ErrorRecoveryService
from src.models.data_models import ContentType


async def test_fallback_content():
    ers = ErrorRecoveryService()

    # Test different content types
    content_types = [
        ContentType.QUALIFICATION,
        ContentType.EXPERIENCE,
        ContentType.PROJECT,
        ContentType.EXECUTIVE_SUMMARY,
    ]

    for content_type in content_types:
        content = await ers.get_fallback_content(
            content_type=content_type,
            error_message="Test error",
            item_id="test_item",
            field="software development",
        )
        print(f"{content_type.value}: {content}")


if __name__ == "__main__":
    asyncio.run(test_fallback_content())
