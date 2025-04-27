import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
from vector_db import VectorDB
from state_manager import VectorStoreConfig, ExperienceEntry

# Mock classes and functions for faiss and uuid
MockIndexFlatL2 = MagicMock()
MockIndexIVFFlat = MagicMock()
MockUuid4 = MagicMock(return_value="mock-uuid")

class TestVectorDB(unittest.TestCase):

    @patch('vector_db.faiss.IndexFlatL2', new=MockIndexFlatL2)
    @patch('vector_db.faiss.IndexIVFFlat', new=MockIndexIVFFlat)
    @patch('vector_db.uuid4', new=MockUuid4)
    @patch('vector_db.genai.embed_content')
    def setUp(self, mock_embed_content):
        """Set up mock objects and VectorDB instance before each test."""
        self.mock_embed_content = mock_embed_content
        self.mock_embed_content.return_value = {'embedding': [0.1] * 768}

        # Reset mocks before each test
        MockIndexFlatL2.reset_mock()
        MockIndexIVFFlat.reset_mock()
        MockUuid4.reset_mock()
        self.mock_embed_content.reset_mock()

        # Configure mock index instances
        self.mock_index_flat = MagicMock()
        self.mock_index_ivf = MagicMock()
        MockIndexFlatL2.return_value = self.mock_index_flat
        MockIndexIVFFlat.return_value = self.mock_index_ivf
        self.mock_index_flat.is_trained = True # Assume trained for simplicity in most tests
        self.mock_index_ivf.is_trained = True


    @patch('vector_db.faiss.IndexFlatL2', new=MockIndexFlatL2)
    def test_init_index_flat_l2(self):
        """Test initialization with IndexFlatL2."""
        config = VectorStoreConfig(dimension=128, index_type="IndexFlatL2")
        db = VectorDB(config=config)

        MockIndexFlatL2.assert_called_once_with(128)
        self.assertEqual(db.config, config)
        self.assertIsNotNone(db._embed_function) # Should use default embed function

    @patch('vector_db.faiss.IndexFlatL2', new=MockIndexFlatL2) # For quantizer
    @patch('vector_db.faiss.IndexIVFFlat', new=MockIndexIVFFlat)
    def test_init_index_ivf_flat(self):
        """Test initialization with IndexIVFFlat."""
        config = VectorStoreConfig(dimension=128, index_type="IndexIVFFlat")
        db = VectorDB(config=config)

        # Check calls for quantizer and IndexIVFFlat
        quantizer_call = call(128)
        index_ivf_call = call(MockIndexFlatL2.return_value, 128, 100)
        MockIndexFlatL2.assert_has_calls([quantizer_call], any_order=False)
        MockIndexIVFFlat.assert_has_calls([index_ivf_call], any_order=False)

        self.assertEqual(db.config, config)
        self.assertIsNotNone(db._embed_function)

    def test_init_invalid_index_type(self):
        """Test initialization with an invalid index type."""
        config = VectorStoreConfig(dimension=128, index_type="InvalidIndex")
        with self.assertRaisesRegex(Exception, "Invalid index type"):
            VectorDB(config=config)

    def test_init_with_custom_embed_function(self):
        """Test initialization with a custom embed function."""
        config = VectorStoreConfig(dimension=128, index_type="IndexFlatL2")
        mock_custom_embed = MagicMock()
        db = VectorDB(config=config, embed_function=mock_custom_embed)

        self.assertEqual(db._embed_function, mock_custom_embed)

    @patch('vector_db.faiss.IndexFlatL2', new=MockIndexFlatL2)
    @patch('vector_db.uuid4', new=MockUuid4)
    def test_add_item_with_text_attribute(self):
        """Test adding an item with a text attribute."""
        config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
        db = VectorDB(config=config, embed_function=self.mock_embed_content) # Use mock embed

        item_to_add = ExperienceEntry(text="This is an experience.")
        db.add_item(item_to_add)

        # Check if uuid4 was called
        MockUuid4.assert_called_once()
        # Check if the item was stored in data
        self.assertIn("mock-uuid", db.data)
        self.assertEqual(db.data["mock-uuid"], item_to_add)
        # Check if embed function was called with the correct text
        self.mock_embed_content.assert_called_once_with("This is an experience.")
        # Check if index.add was called with the embedding
        expected_embedding = np.array([self.mock_embed_content.return_value['embedding']])
        self.mock_index_flat.add.assert_called_once()
        np.testing.assert_array_equal(self.mock_index_flat.add.call_args[0][0], expected_embedding)
        # Check if index.train was NOT called (assuming index is_trained = True)
        self.mock_index_flat.train.assert_not_called()

    @patch('vector_db.faiss.IndexFlatL2', new=MockIndexFlatL2)
    @patch('vector_db.uuid4', new=MockUuid4)
    def test_add_item_with_explicit_text(self):
        """Test adding an item with explicit text provided."""
        config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
        db = VectorDB(config=config, embed_function=self.mock_embed_content)

        item_to_add = {"some_key": "some_value"} # No text attribute
        explicit_text = "Explicit text for embedding."
        db.add_item(item_to_add, text=explicit_text)

        MockUuid4.assert_called_once()
        self.assertIn("mock-uuid", db.data)
        self.assertEqual(db.data["mock-uuid"], item_to_add)
        self.mock_embed_content.assert_called_once_with(explicit_text)

        expected_embedding = np.array([self.mock_embed_content.return_value['embedding']])
        self.mock_index_flat.add.assert_called_once()
        np.testing.assert_array_equal(self.mock_index_flat.add.call_args[0][0], expected_embedding)
        self.mock_index_flat.train.assert_not_called()

    @patch('vector_db.faiss.IndexFlatL2', new=MockIndexFlatL2)
    @patch('vector_db.uuid4', new=MockUuid4)
    def test_add_item_requires_text(self):
        """Test that add_item raises error if no text is available."""
        config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
        db = VectorDB(config=config, embed_function=self.mock_embed_content)

        item_to_add = {"some_key": "some_value"} # No text attribute and no text provided
        with self.assertRaisesRegex(Exception, "The item must have a text attribute"):
            db.add_item(item_to_add)

        MockUuid4.assert_not_called()
        self.mock_embed_content.assert_not_called()
        self.mock_index_flat.add.assert_not_called()

    @patch('vector_db.faiss.IndexFlatL2', new=MockIndexFlatL2)
    @patch('vector_db.uuid4', new=MockUuid4)
    def test_add_item_trains_index_if_needed(self):
        """Test that add_item trains the index if it's not trained."""
        config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
        db = VectorDB(config=config, embed_function=self.mock_embed_content)
        self.mock_index_flat.is_trained = False # Simulate untrained index

        item_to_add = ExperienceEntry(text="Another experience.")
        db.add_item(item_to_add)

        self.mock_index_flat.train.assert_called_once()
        # Verify train was called with the correct embedding format
        train_call_arg = self.mock_index_flat.train.call_args[0][0]
        self.assertIsInstance(train_call_arg, np.ndarray)
        self.assertEqual(train_call_arg.shape, (1, 768))

        self.mock_index_flat.add.assert_called_once()

    @patch('vector_db.faiss.IndexFlatL2', new=MockIndexFlatL2)
    @patch('vector_db.uuid4', new=MockUuid4)
    def test_search(self):
        """Test the search method."""
        config = VectorStoreConfig(dimension=768, index_type="IndexFlatL2")
        db = VectorDB(config=config, embed_function=self.mock_embed_content)

        # Add some dummy data to the internal data dictionary
        item1 = ExperienceEntry(text="Item 1")
        item2 = ExperienceEntry(text="Item 2")
        item3 = ExperienceEntry(text="Item 3")
        db.data = {"id1": item1, "id2": item2, "id3": item3}

        # Configure the mock index search method
        # Simulate search returning indices corresponding to item2, item1, item3
        self.mock_index_flat.search.return_value = (np.array([[0.5, 0.6, 0.7]]), np.array([[1, 0, 2]]))
        # Note: The indices returned by faiss.search correspond to the indices in the order data was added to the faiss index.
        # However, our current VectorDB implementation maps these indices back to the keys in self.data
        # based on the *order* of keys in the dictionary. This is fragile.
        # A more robust implementation would store a mapping from faiss index id to our internal data id.
        # For this test, we'll simulate the current logic: map faiss index to internal data key order.

        # Let's fix the mock search return value to match the expected behavior based on dictionary key order.
        # If data was added in the order id1, id2, id3, faiss index 0 corresponds to id1, 1 to id2, 2 to id3.
        # Let's assume the search returns indices [1, 0, 2] which should map to [id2, id1, id3].
        self.mock_index_flat.search.return_value = (np.array([[0.5, 0.6, 0.7]]), np.array([[1, 0, 2]]))

        query_text = "Search query."
        k_value = 3
        results = db.search(query_text, k=k_value)

        # Check if embed function was called for the query
        self.mock_embed_content.assert_called_once_with(query_text)

        # Check if index.search was called correctly
        expected_query_embed = np.array([self.mock_embed_content.return_value['embedding']])
        self.mock_index_flat.search.assert_called_once()
        np.testing.assert_array_equal(self.mock_index_flat.search.call_args[0][0], expected_query_embed)
        self.assertEqual(self.mock_index_flat.search.call_args[0][1], k_value)

        # Check the returned results based on the mock search result and data mapping
        # Expected order: item2, item1, item3
        self.assertEqual(len(results), k_value)
        self.assertEqual(results[0], item2)
        self.assertEqual(results[1], item1)
        self.assertEqual(results[2], item3)

if __name__ == '__main__':
    unittest.main()
