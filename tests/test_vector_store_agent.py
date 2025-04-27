import unittest
from unittest.mock import MagicMock, call
from vector_store_agent import VectorStoreAgent
from state_manager import VectorStoreConfig, AgentIO
from agent_base import AgentBase # Import AgentBase to define a concrete subclass

# Define a concrete subclass of VectorStoreAgent for testing
class ConcreteVectorStoreAgent(VectorStoreAgent):
    def run(self, input: any) -> any:
        """Concrete implementation of run for testing."""
        # This method is not under test in this file, so a placeholder is sufficient.
        pass

class TestVectorStoreAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and ConcreteVectorStoreAgent instance before each test."""
        self.mock_vector_db = MagicMock()
        self.mock_config = VectorStoreConfig(dimension=100, index_type="IndexFlatL2")
        # Instantiate the concrete subclass instead of the abstract base class
        self.agent = ConcreteVectorStoreAgent(
            name="TestVectorStoreAgent",
            description="Test Description",
            vector_store=self.mock_vector_db,
            config=self.mock_config
        )

    def test_init(self):
        """Test that VectorStoreAgent (via ConcreteVectorStoreAgent) is initialized correctly."""
        # Note: We are testing the __init__ inherited from VectorStoreAgent
        self.assertEqual(self.agent.name, "TestVectorStoreAgent")
        self.assertEqual(self.agent.description, "Test Description")
        self.assertEqual(self.agent.vector_store, self.mock_vector_db)
        self.assertEqual(self.agent.config, self.mock_config)
        self.assertIsInstance(self.agent.input_schema, AgentIO)
        self.assertIsInstance(self.agent.output_schema, AgentIO)
        # Check schema content - Note: schemas are hardcoded in __init__
        self.assertEqual(self.agent.input_schema["input"], {})
        self.assertEqual(self.agent.input_schema["output"], {})
        self.assertEqual(self.agent.input_schema["description"], "Agent for managing a vector store.")
        self.assertEqual(self.agent.output_schema["input"], {})
        self.assertEqual(self.agent.output_schema["output"], {})
        self.assertEqual(self.agent.output_schema["description"], "Agent for managing a vector store.")

    def test_run_add_item(self):
        """Test the run_add_item method."""
        item_to_add = {"data": "some data"}
        text_for_embedding = "text to embed"

        self.agent.run_add_item(item_to_add, text=text_for_embedding)

        # Assert that vector_store.add_item was called with the correct arguments
        self.mock_vector_db.add_item.assert_called_once_with(item_to_add, text_for_embedding)

    def test_run_search(self):
        """Test the run_search method."""
        query_text = "search query"
        k_value = 5
        mock_search_results = [{"result": "found item"}]

        # Configure the mock vector_db.search to return a specific value
        self.mock_vector_db.search.return_value = mock_search_results

        results = self.agent.run_search(query_text, k=k_value)

        # Assert that vector_store.search was called with the correct arguments
        self.mock_vector_db.search.assert_called_once_with(query_text, k_value)

        # Assert that the method returned the result from vector_store.search
        self.assertEqual(results, mock_search_results)

if __name__ == '__main__':
    unittest.main()
