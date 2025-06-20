"""Unit tests for verifying correct executor usage in LLM service."""

import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import concurrent.futures

from src.services.llm_service import EnhancedLLMService
from src.config.settings import get_config


class TestExecutorUsage(unittest.TestCase):
    """Test that LLM service uses self.executor instead of None."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the config to avoid dependency issues
        with patch('src.services.llm_service.get_config') as mock_config:
            mock_config.return_value = Mock(
                llm=Mock(
                    gemini_api_key_primary="test_key",
                    gemini_api_key_fallback=None
                ),
                llm_settings=Mock(default_model="gemini-pro")
            )
            
            # Mock genai to avoid actual API calls
            with patch('src.services.llm_service.genai') as mock_genai:
                mock_genai.GenerativeModel.return_value = Mock()
                mock_genai.configure = Mock()
                
                # Mock other dependencies
                with patch('src.services.llm_service.get_advanced_cache') as mock_cache:
                    mock_cache.return_value = Mock()
                    
                    with patch('src.services.llm_service.get_performance_optimizer') as mock_perf:
                        mock_perf.return_value = Mock()
                        
                        with patch('src.services.llm_service.get_async_optimizer') as mock_async:
                            mock_async.return_value = Mock()
                            
                            with patch('src.services.llm_service.get_error_recovery_service') as mock_error:
                                mock_error.return_value = Mock()
                                
                                self.llm_service = EnhancedLLMService()

    def test_executor_initialization(self):
        """Test that self.executor is properly initialized."""
        self.assertIsInstance(
            self.llm_service.executor, 
            concurrent.futures.ThreadPoolExecutor
        )
        self.assertEqual(self.llm_service.executor._max_workers, 5)
        self.assertTrue(
            self.llm_service.executor._thread_name_prefix.startswith("llm_worker")
        )

    @patch('asyncio.get_event_loop')
    async def test_run_in_executor_uses_self_executor(self, mock_get_loop):
        """Test that _generate_with_timeout uses self.executor."""
        # Setup mock loop
        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop
        
        # Mock the executor task
        mock_future = asyncio.Future()
        mock_future.set_result("test_result")
        mock_loop.run_in_executor.return_value = mock_future
        
        # Mock the _make_llm_api_call method
        self.llm_service._make_llm_api_call = Mock(return_value="test_result")
        
        # Call the method
        result = await self.llm_service._generate_with_timeout("test prompt")
        
        # Verify that run_in_executor was called with self.executor
        mock_loop.run_in_executor.assert_called_once_with(
            self.llm_service.executor,
            self.llm_service._make_llm_api_call,
            "test prompt"
        )
        
        # Verify the result
        self.assertEqual(result, "test_result")

    def test_executor_not_none(self):
        """Test that executor is not None."""
        self.assertIsNotNone(self.llm_service.executor)
        self.assertIsInstance(
            self.llm_service.executor,
            concurrent.futures.ThreadPoolExecutor
        )


if __name__ == '__main__':
    unittest.main()