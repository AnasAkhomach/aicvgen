import chromadb
from ..utils.exceptions import VectorStoreError, ConfigurationError
from ..config.settings import get_config
from typing import List, Dict, Any, Optional
import logging
import hashlib
from ..models.vector_store_and_session_models import VectorStoreSearchResult

logger = logging.getLogger("vector_store_service")


class VectorStoreService:
    def __init__(self, settings=None):
        self.settings = settings or get_config()
        self.client = self._connect()
        self.collection = self._get_or_create_collection()

    def _connect(self):
        try:
            client = chromadb.PersistentClient(
                path=self.settings.vector_db.persist_directory
            )
            # Optionally, ping or check connection
            if hasattr(client, "heartbeat"):
                client.heartbeat()
            logger.info(
                f"Successfully connected to ChromaDB at {self.settings.vector_db.persist_directory}"
            )
            return client
        except Exception as e:
            raise ConfigurationError(
                f"CRITICAL: Failed to connect to Vector Store (ChromaDB) at path '{self.settings.vector_db.persist_directory}'. "
                f"Please check the path and permissions. Error: {e}"
            ) from e

    def _get_or_create_collection(self, collection_name: str = "cv_content"):
        """Get or create a ChromaDB collection for storing CV content."""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "CV content and job description data"},
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to get or create collection '{collection_name}': {e}")
            raise ConfigurationError(f"Failed to initialize collection: {e}") from e

    def get_client(self):
        return self.client

    def add_item(
        self, item: Any, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add an item to the vector store.

        Args:
            item: The item object (for compatibility with existing code)
            content: The text content to store
            metadata: Optional metadata dictionary

        Returns:
            str: The ID of the stored item
        """
        try:
            # Generate a unique ID for the item
            item_id = self._generate_id(content)

            # Prepare metadata
            meta = metadata or {}
            if hasattr(item, "__dict__"):
                # Add item attributes to metadata if it's an object
                meta.update(
                    {
                        k: str(v)
                        for k, v in item.__dict__.items()
                        if isinstance(v, (str, int, float, bool))
                    }
                )

            # Add to ChromaDB collection
            self.collection.add(documents=[content], metadatas=[meta], ids=[item_id])

            logger.debug(f"Added item to vector store with ID: {item_id}")
            return item_id

        except Exception as e:
            logger.error(f"Failed to add item to vector store: {e}")
            raise

    def search(self, query: str, k: int = 5) -> list[VectorStoreSearchResult]:
        """Search for similar content in the vector store.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List[VectorStoreSearchResult]: List of search results with content and metadata
        """
        try:
            results = self.collection.query(query_texts=[query], n_results=k)
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    result = VectorStoreSearchResult(
                        content=doc,
                        metadata=(
                            results["metadatas"][0][i]
                            if results["metadatas"] and results["metadatas"][0]
                            else {}
                        ),
                        id=(
                            results["ids"][0][i]
                            if results["ids"] and results["ids"][0]
                            else None
                        ),
                        distance=(
                            results["distances"][0][i]
                            if results["distances"] and results["distances"][0]
                            else None
                        ),
                    )
                    formatted_results.append(result)

            logger.debug(
                f"Search returned {len(formatted_results)} results for query: {query[:50]}..."
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content based on its hash."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()


_vector_store_instance = None


def get_vector_store_service():
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
    return _vector_store_instance
