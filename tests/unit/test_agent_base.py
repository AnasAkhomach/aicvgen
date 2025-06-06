import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
from unittest.mock import MagicMock, patch
from src.agents.agent_base import AgentBase
from src.core.state_manager import AgentIO
from typing import Any  # Import Any


# Create a concrete implementation of AgentBase for testing
class ConcreteAgent(AgentBase):
    def __init__(
        self, name: str, description: str, input_schema: AgentIO, output_schema: AgentIO
    ):
        super().__init__(name, description, input_schema, output_schema)

    def run(self, input: Any) -> Any:
        # This will be mocked in tests, or we can raise NotImplementedError here
        # For testing the base class, we'll focus on the methods defined in AgentBase
        pass


class TestAgentBase(unittest.TestCase):

    def setUp(self):
        """Set up a concrete agent instance before each test."""
        self.input_schema = AgentIO(
            input={"text": str}, output={}, description="Input schema"
        )
        self.output_schema = AgentIO(
            input={}, output={"processed_text": str}, description="Output schema"
        )
        self.agent = ConcreteAgent(
            name="TestAgent",
            description="A test agent for AgentBase.",
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )

    def test_init(self):
        """Test that AgentBase is initialized correctly."""
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertEqual(self.agent.description, "A test agent for AgentBase.")
        self.assertEqual(self.agent.input_schema, self.input_schema)
        self.assertEqual(self.agent.output_schema, self.output_schema)

    def test_run_abstract_method(self):
        """Test that calling the abstract run method raises NotImplementedError."""
        # Although our concrete agent has a pass in run, if it didn't, this is how you'd test the abstract nature
        # However, testing the abstract method itself is done by ensuring concrete classes implement it.
        # A better test for the abstract concept is to ensure AgentBase cannot be instantiated directly.
        with self.assertRaises(TypeError):
            AgentBase(
                name="Abstract",
                description="Abstract",
                input_schema=self.input_schema,
                output_schema=self.output_schema,
            )

    @patch("agent_base.logging.info")
    def test_log_decision(self, mock_log_info):
        """Test that log_decision calls logging.info with the correct format."""
        message = "Decision made."
        self.agent.log_decision(message)
        mock_log_info.assert_called_once_with("[%s] %s", self.agent.name, message)

    def test_generate_explanation(self):
        """Test that generate_explanation produces the correct output string."""
        input_data = {"text": "some input"}
        output_data = {"processed_text": "some output"}
        expected_explanation = (
            f"Agent '{self.agent.name}' processed the input data and generated the following output:\n"
            f"Input: {input_data}\n"
            f"Output: {output_data}\n"
            f"Description: {self.agent.description}"
        )
        explanation = self.agent.generate_explanation(input_data, output_data)
        self.assertEqual(explanation, expected_explanation)

    def test_get_confidence_score(self):
        """Test that the default get_confidence_score returns 1.0."""
        output_data = {"processed_text": "some output"}
        confidence = self.agent.get_confidence_score(output_data)
        self.assertEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
