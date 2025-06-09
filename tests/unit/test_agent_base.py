import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.agents.agent_base import EnhancedAgentBase, AgentExecutionContext, AgentResult
from src.core.state_manager import AgentIO
from src.models.data_models import ContentType
from typing import Any


# Create a concrete implementation of EnhancedAgentBase for testing
class ConcreteEnhancedAgent(EnhancedAgentBase):
    def __init__(
        self, name: str, description: str, input_schema: AgentIO, output_schema: AgentIO, content_type: ContentType = None
    ):
        super().__init__(name, description, input_schema, output_schema, content_type)

    async def run_async(self, input_data: Any, context: AgentExecutionContext) -> AgentResult:
        # Mock implementation for testing
        return AgentResult(
            success=True,
            output_data={"processed": input_data},
            confidence_score=0.9,
            processing_time=0.1
        )
    
    def run(self, input_data: Any) -> Any:
        # Legacy method implementation for testing
        return {"processed": input_data}


class TestEnhancedAgentBase(unittest.TestCase):

    def setUp(self):
        """Set up a concrete enhanced agent instance before each test."""
        self.input_schema = AgentIO(
            input={"text": str}, output={}, description="Input schema"
        )
        self.output_schema = AgentIO(
            input={}, output={"processed_text": str}, description="Output schema"
        )
        
        # Mock the enhanced services to avoid actual service initialization
        with patch('src.agents.agent_base.get_structured_logger'), \
             patch('src.agents.agent_base.get_error_recovery_service'), \
             patch('src.agents.agent_base.get_progress_tracker'), \
             patch('src.agents.agent_base.get_session_manager'):
            
            self.agent = ConcreteEnhancedAgent(
                name="TestEnhancedAgent",
                description="A test agent for EnhancedAgentBase.",
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                content_type=ContentType.EXPERIENCE
            )

    def test_init(self):
        """Test that EnhancedAgentBase is initialized correctly."""
        self.assertEqual(self.agent.name, "TestEnhancedAgent")
        self.assertEqual(self.agent.description, "A test agent for EnhancedAgentBase.")
        self.assertEqual(self.agent.input_schema, self.input_schema)
        self.assertEqual(self.agent.output_schema, self.output_schema)
        self.assertEqual(self.agent.content_type, ContentType.EXPERIENCE)
        
        # Check that performance tracking attributes are initialized
        self.assertEqual(self.agent.execution_count, 0)
        self.assertEqual(self.agent.total_processing_time, 0.0)
        self.assertEqual(self.agent.success_count, 0)
        self.assertEqual(self.agent.error_count, 0)

    def test_abstract_instantiation(self):
        """Test that EnhancedAgentBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            EnhancedAgentBase(
                name="Abstract",
                description="Abstract",
                input_schema=self.input_schema,
                output_schema=self.output_schema,
            )

    def test_async_run_method(self):
        """Test the async run_async method."""
        async def run_test():
            input_data = {"text": "test input"}
            context = AgentExecutionContext(
                session_id="test-session",
                item_id="test-item",
                content_type=ContentType.EXPERIENCE
            )
            
            result = await self.agent.run_async(input_data, context)
            
            self.assertIsInstance(result, AgentResult)
            self.assertTrue(result.success)
            self.assertEqual(result.output_data, {"processed": input_data})
            self.assertEqual(result.confidence_score, 0.9)
            self.assertEqual(result.processing_time, 0.1)
        
        asyncio.run(run_test())

    def test_agent_execution_context(self):
        """Test AgentExecutionContext dataclass."""
        context = AgentExecutionContext(
            session_id="test-session",
            item_id="test-item",
            content_type=ContentType.QUALIFICATION,
            retry_count=2
        )
        
        self.assertEqual(context.session_id, "test-session")
        self.assertEqual(context.item_id, "test-item")
        self.assertEqual(context.content_type, ContentType.QUALIFICATION)
        self.assertEqual(context.retry_count, 2)
        self.assertIsInstance(context.metadata, dict)
        self.assertIsInstance(context.input_data, dict)
        self.assertIsInstance(context.processing_options, dict)

    def test_agent_result(self):
        """Test AgentResult dataclass."""
        result = AgentResult(
            success=True,
            output_data={"key": "value"},
            confidence_score=0.85,
            processing_time=1.5,
            error_message=None
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.output_data, {"key": "value"})
        self.assertEqual(result.confidence_score, 0.85)
        self.assertEqual(result.processing_time, 1.5)
        self.assertIsNone(result.error_message)
        self.assertIsInstance(result.metadata, dict)

    def test_agent_result_with_error(self):
        """Test AgentResult with error information."""
        result = AgentResult(
            success=False,
            output_data=None,
            confidence_score=0.0,
            error_message="Test error occurred"
        )
        
        self.assertFalse(result.success)
        self.assertIsNone(result.output_data)
        self.assertEqual(result.confidence_score, 0.0)
        self.assertEqual(result.error_message, "Test error occurred")


if __name__ == "__main__":
    unittest.main()
