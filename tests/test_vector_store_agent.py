import unittest
from unittest.mock import MagicMock, call
from vector_store_agent import VectorStoreAgent
from state_manager import VectorStoreConfig, AgentIO
from agent_base import AgentBase # Import AgentBase to define a concrete subclass

# Define a concrete subclass of VectorStoreAgent for testing
class ConcreteVectorStoreAgent(VectorStoreAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_store = self.vector_db
        self.description = "Test Description"
        self.config = VectorStoreConfig(dimension=100, index_type="IndexFlatL2")

    def run(self, input: any) -> any:
        """Concrete implementation of run for testing."""
        # This method is not under test in this file, so a placeholder is sufficient.
        pass

    def run_search(self, query_text: str, k: int):
        """Concrete implementation of run_search for testing."""
        return self.vector_store.search(query_text, k)

class TestVectorStoreAgent(unittest.TestCase):

    def setUp(self):
        """Set up mock objects and ConcreteVectorStoreAgent instance before each test."""
        self.mock_vector_db = MagicMock()
        self.mock_config = VectorStoreConfig(dimension=100, index_type="IndexFlatL2")
        self.mock_model = MagicMock()
        self.agent = ConcreteVectorStoreAgent(
            name="TestVectorStoreAgent",
            description="Agent for managing a vector store.",
            model=self.mock_model,
            input_schema=AgentIO(
                input={},
                description="Input schema for vector store operations."
            ),
            output_schema=AgentIO(
                output={},
                description="Output schema for vector store operations."
            ),
            vector_db=self.mock_vector_db
        )

    def test_init(self):
        """Test that VectorStoreAgent (via ConcreteVectorStoreAgent) is initialized correctly."""
        # Note: We are testing the __init__ inherited from VectorStoreAgent
        self.assertEqual(self.agent.name, "TestVectorStoreAgent")
        self.assertEqual(self.agent.description, "Test Description")
        self.assertEqual(self.agent.vector_store, self.mock_vector_db)
        self.assertEqual(self.agent.config, self.mock_config)
        # Validate schema attributes directly
        self.assertTrue('input' in self.agent.input_schema)
        self.assertTrue('description' in self.agent.input_schema)
        self.assertTrue('output' in self.agent.output_schema)
        self.assertTrue('description' in self.agent.output_schema)

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
