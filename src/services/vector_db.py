from typing import List, Dict, Any, Callable, Optional
import faiss
import numpy as np
import google.generativeai as genai
from src.core.state_manager import VectorStoreConfig, SkillEntry, ExperienceEntry
from uuid import uuid4
import logging

# Set up logging
logger = logging.getLogger(__name__)


class VectorDB:
    """
    Simple in-memory vector database for skills and experiences.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        embed_function: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initializes the VectorDB with the given configuration.

        Args:
            config: The VectorStoreConfig object.
            embed_function: Optional. A function to generate embeddings.
                             If None, uses a default Gemini embedding function.
        """
        if config.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(config.dimension)
        elif config.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(config.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, config.dimension, 100)
        else:
            raise Exception("Invalid index type")

        self.data: Dict[str, Any] = {}  # Store data items with ids as keys
        self.metadata: Dict[str, Dict[str, Any]] = {}  # Store metadata with ids as keys
        self.id_map: Dict[int, str] = {}  # Map index positions to ids
        self.next_index = 0  # Track the next index position
        self.config = config
        self.indexed_cv_ids = set()  # Track which CV IDs have been indexed
        self._embed_function = (
            embed_function if embed_function is not None else self._default_embed_function
        )

    def _default_embed_function(self, text: str) -> List[float]:
        """
        Default embedding function using Gemini.

        Args:
            text: The text to embed.

        Returns:
            The embedding of the text
        """
        try:
            result = genai.embed_content(model="models/embedding-001", content=text)
            return result["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * self.config.dimension

    def add_item(self, item: Any, text: str = None, metadata: Dict[str, Any] = None):
        """
        Adds a item to the vector database.

        Args:
            item: The item to add.
            text: The text to use for generating the embedding. If None, it will use item.text.
            metadata: Optional metadata to store with the item.
        """
        if text is None:
            if hasattr(item, "text"):
                text = item.text
            elif hasattr(item, "content"):
                text = item.content
            else:
                raise Exception("The item must have a text or content attribute")

        # Generate a unique ID if one isn't provided in metadata
        item_id = metadata.get("id", str(uuid4())) if metadata else str(uuid4())

        # Store the item and metadata
        self.data[item_id] = item
        self.metadata[item_id] = metadata or {}

        try:
            # Generate embedding
            embedding = self._embed_function(text)

            # Train the index if needed
            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                self.index.train(np.array([embedding], dtype=np.float32))

            # Add to the index
            self.index.add(np.array([embedding]))

            # Map the index position to the ID
            self.id_map[self.next_index] = item_id
            self.next_index += 1

            logger.debug(f"Added item with ID {item_id} to vector DB")

        except Exception as e:
            logger.error(f"Error adding item to vector database: {str(e)}")

    def search(self, query: str, k=3) -> List[Any]:
        """
        Searches for the k most similar items to the query.

        Args:
            query: The query to search for.
            k: The number of results to return.

        Returns:
            List of similar items with their metadata attached.
        """
        if not self.data:
            logger.warning("Search attempted on empty vector database")
            return []

        try:
            # Generate query embedding
            query_embed = self._embed_function(query)

            # Search the index
            distances, indices = self.index.search(np.array([query_embed]), k)

            # Process the results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= self.next_index:
                    continue  # Invalid index

                item_id = self.id_map.get(idx)
                if not item_id or item_id not in self.data:
                    continue  # Missing ID mapping

                item = self.data[item_id]

                # Attach metadata to the item
                if hasattr(item, "__dict__"):
                    # For class instances, add metadata as attributes
                    for key, value in self.metadata.get(item_id, {}).items():
                        setattr(item, key, value)

                    # Add the distance as a relevance score
                    setattr(item, "relevance_score", float(distances[0][i]))

                results.append(item)

            return results

        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []

    def clear(self):
        """
        Clears the vector database.
        """
        self.data = {}
        self.metadata = {}
        self.id_map = {}
        self.next_index = 0
        self.indexed_cv_ids = set()

        # Reset the index
        if self.config.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.config.dimension)
        elif self.config.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(self.config.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.config.dimension, 100)
            # Make sure it's properly initialized for new data
            self.index.train(np.zeros((1, self.config.dimension), dtype=np.float32))

        logger.info("Vector database cleared")
