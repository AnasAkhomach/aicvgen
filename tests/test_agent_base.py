import unittest
from unittest.mock import MagicMock
from agent_base import AgentBase

# We will test the abstract nature by attempting to instantiate subclasses without implementing abstract methods

class TestAgentBase(unittest.TestCase):

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that AgentBase cannot be instantiated directly."""
        name = "TestAgent"
        description = "A test agent"
        input_schema = MagicMock()
        output_schema = MagicMock()

        with self.assertRaises(TypeError) as cm:
            AgentBase(name, description, input_schema, output_schema)

        # Verify the error message indicates an abstract method is missing
        self.assertIn("Can't instantiate abstract class AgentBase with abstract method run", str(cm.exception))

    def test_subclass_without_implementing_run_cannot_be_instantiated(self):
        """Test that a subclass not implementing run cannot be instantiated."""
        # Define a subclass that inherits from AgentBase but does not implement run
        class IncompleteAgent(AgentBase):
            pass

        name = "Incomplete"
        description = "An incomplete agent"
        input_schema = MagicMock()
        output_schema = MagicMock()

        with self.assertRaises(TypeError) as cm:
            IncompleteAgent(name, description, input_schema, output_schema)

        # Verify the error message indicates the abstract method is missing
        self.assertIn("Can't instantiate abstract class IncompleteAgent with abstract method run", str(cm.exception))

    def test_concrete_subclass_initialization(self):
        """Test that a concrete subclass can be initialized and inherits base properties."""
        # Define a concrete subclass that implements the abstract method run
        class ConcreteAgent(AgentBase):
            def run(self, input: any) -> any:
                return "Concrete result"

        name = "ConcreteTestAgent"
        description = "A concrete test agent"
        input_schema = MagicMock()
        output_schema = MagicMock()

        agent = ConcreteAgent(name, description, input_schema, output_schema)

        self.assertEqual(agent.name, name)
        self.assertEqual(agent.description, description)
        self.assertEqual(agent.input_schema, input_schema)
        self.assertEqual(agent.output_schema, output_schema)
        # Optionally, test the implemented run method
        self.assertEqual(agent.run("some input"), "Concrete result")

if __name__ == '__main__':
    unittest.main()
