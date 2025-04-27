from typing import List, Dict, Any, Callable, Optional
import faiss
import numpy as np
import google.generativeai as genai 
from state_manager import VectorStoreConfig, SkillEntry, ExperienceEntry
from uuid import uuid4


class VectorDB:
    """
    Simple in-memory vector database for skills and experiences.
    """

    def __init__(self, config: VectorStoreConfig, embed_function: Optional[Callable[[str], List[float]]] = None):
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

        self.data: Dict[str, Any] = {}  # Store data with ids as keys
        self.config = config
        self._embed_function = embed_function if embed_function is not None else self._default_embed_function

    def _default_embed_function(self, text: str) -> List[float]:
        """
        Default embedding function using Gemini.

        Args:
            text: The text to embed.

        Returns:
            The embedding of the text
        """
        return genai.embed_content(model='models/embedding-001', content=text)['embedding']

    def add_item(self, item: Any, text: str = None):
        """
        Adds a item to the vector database.

        Args:
            item: The item to add.
            text: The text to use for generating the embedding. If None, it will use item.text.
        """
        if text is None:
            if hasattr(item, "text"):
                text = item.text
            else:
                raise Exception("The item must have a text attribute")

        id = str(uuid4())
        self.data[id] = item
        embedding = self._embed_function(text)
        if not self.index.is_trained:  # Check if the index needs training
            self.index.train(np.array([embedding]))  # Train the index
        self.index.add(np.array([embedding]))

    def search(self, query: str, k=3) -> List[Any]:
        """
        Searches for the k most similar items to the query.

        Args:
            query: The query to search for.
            k: The number of results to return.

        Returns:
            List of similar items.
        """
        query_embed = self._embed_function(query)
        _, indices = self.index.search(np.array([query_embed]), k)
        ids = [list(self.data.keys())[i] for i in indices[0]]
        return [self.data[id] for id in ids]