import chromadb
from ..error_handling.exceptions import (
    VectorStoreError,
    ConfigurationError,
    OperationTimeoutError,
)
from ..config.settings import get_config
from typing import List, Dict, Any, Optional
import logging
import hashlib
import threading
import time
from ..models.vector_store_and_session_models import VectorStoreSearchResult

logger = logging.getLogger("vector_store_service")


def run_with_timeout(func, args=(), kwargs=None, timeout=30):
    """Run a function with a timeout using threading."""
    if kwargs is None:
        kwargs = {}

    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except BaseException as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise OperationTimeoutError(f"Operation timed out after {timeout} seconds")

    if exception[0] is not None:
        raise exception[0]
    else:
        raise Exception("An unknown error occurred in the timed operation.")

    return result[0]


class VectorStoreService:
    def __init__(self, settings):
        self.settings = settings
        self.client = self._connect()
        self.collection = self._get_or_create_collection()

    def shutdown(self):
        """Shutdown the vector store service and release resources."""
        logger.info("Shutting down VectorStoreService.")
        self.client = None
        self.collection = None
        logger.info("VectorStoreService shutdown complete.")

    def _connect(self):
        try:
            logger.info(
                "Initializing ChromaDB at %s", self.settings.vector_db.persist_directory
            )

            # Create directory if it doesn't exist
            import os

            os.makedirs(self.settings.vector_db.persist_directory, exist_ok=True)

            # Try to connect with a timeout to prevent hanging
            def create_client():
                client = chromadb.PersistentClient(
                    path=self.settings.vector_db.persist_directory
                )
                # Test the connection with a simple operation
                logger.info("Testing ChromaDB connection...")
                test_collection = client.get_or_create_collection("test_connection")
                # Try a simple operation to ensure it's working
                test_collection.count()
                logger.info("ChromaDB connection test successful")
                return client

            try:
                client = run_with_timeout(create_client, timeout=30)
            except OperationTimeoutError as e:
                raise ConfigurationError(
                    f"ChromaDB initialization timed out after 30 seconds. "
                    f"This might be due to:\n"
                    f"1. ChromaDB downloading embedding models on first run\n"
                    f"2. Network connectivity issues\n"
                    f"3. File system permission problems\n"
                    f"4. Another process using the database\n\n"
                    f"Try restarting the application or clearing the vector_db directory."
                ) from e

            logger.info(
                "Successfully connected to ChromaDB at %s",
                self.settings.vector_db.persist_directory,
            )
            return client

        except (chromadb.errors.ChromaError, IOError) as e:
            error_msg = (
                f"CRITICAL: Failed to connect to Vector Store (ChromaDB) at path "
                f"'{self.settings.vector_db.persist_directory}'. "
                f"Error: {e}\n\n"
                f"Troubleshooting steps:\n"
                f"1. Check if the directory exists and is writable\n"
                f"2. Ensure you have internet connectivity (ChromaDB may download models)\n"
                f"3. Try clearing the vector_db directory and restart\n"
                f"4. Check for conflicting processes using the database"
            )
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _get_or_create_collection(self, collection_name: str = "cv_content"):
        """Get or create a ChromaDB collection for storing CV content."""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "CV content and job description data"},
            )
            return collection
        except (chromadb.errors.ChromaError, OperationTimeoutError) as e:
            logger.error(
                "Failed to get or create collection '%s': %s", collection_name, e
            )
            raise ConfigurationError(f"Failed to initialize collection: {e}") from e

    def get_client(self):
        return self.client

    def add_item(
        self, item: Any, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add an item to the vector store."""
        try:
            item_id = self._generate_id(content)
            meta = metadata or {}
            if hasattr(item, "__dict__"):
                meta.update(
                    {
                        k: str(v)
                        for k, v in item.__dict__.items()
                        if isinstance(v, (str, int, float, bool))
                    }
                )
            self.collection.add(documents=[content], metadatas=[meta], ids=[item_id])
            logger.debug("Added item to vector store with ID: %s", item_id)
            return item_id
        except (chromadb.errors.ChromaError, TypeError) as e:
            logger.error("Failed to add item to vector store: %s", e)
            raise VectorStoreError("Failed to add item to vector store") from e

    def search(self, query: str, k: int = 5) -> list[VectorStoreSearchResult]:
        """Search for similar content in the vector store."""
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
                "Search returned %d results for query: %s...",
                len(formatted_results),
                query[:50],
            )
            return formatted_results
        except (chromadb.errors.ChromaError, TypeError) as e:
            logger.error("Failed to search vector store: %s", e)
            return []

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content based on its hash."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()


_vector_store_instance = None
_vector_store_lock = threading.Lock()


def get_vector_store_service():
    """Get the global VectorStoreService instance (singleton)."""
    global _vector_store_instance
    if _vector_store_instance is None:
        with _vector_store_lock:
            if _vector_store_instance is None:
                import os

                if os.getenv("SKIP_VECTOR_STORE", "false").lower() == "true":
                    logger.warning(
                        "Skipping vector store initialization (SKIP_VECTOR_STORE=true)"
                    )
                    _vector_store_instance = MockVectorStoreService()
                else:
                    _vector_store_instance = VectorStoreService(get_config())
    return _vector_store_instance


class MockVectorStoreService:
    """Mock vector store service for development/testing."""

    def __init__(self):
        logger.info("Using mock vector store service")

    def add_item(
        self, item: Any, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Mock add_item implementation."""
        return "mock_id"

    def search(self, query: str, k: int = 5) -> list:
        """Mock search implementation."""
        return []

    def get_client(self):
        """Mock get_client implementation."""
        return None

    def shutdown(self):
        """Mock shutdown implementation."""
        logger.info("MockVectorStoreService shutdown.")
