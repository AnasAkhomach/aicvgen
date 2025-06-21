import sys
import os
import asyncio
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.services.llm_service import EnhancedLLMService


@pytest.mark.asyncio
async def test_generate_with_timeout_uses_executor():
    service = EnhancedLLMService()
    loop = asyncio.get_running_loop()
    with patch.object(loop, "run_in_executor", wraps=loop.run_in_executor) as mock_run:
        service.executor = MagicMock()
        # Patch the sync method to return a dummy result
        service._make_llm_api_call = MagicMock(return_value=MagicMock(text="ok"))
        service.timeout = 1
        await service._generate_with_timeout("prompt")
        # Check that run_in_executor was called with service.executor
        assert any(
            call_args[0][0] is service.executor for call_args in mock_run.call_args_list
        )
