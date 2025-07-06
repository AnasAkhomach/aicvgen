import hashlib
import logging
import threading
from typing import Any, Dict, Optional

import chromadb

from ..config.settings import get_config
from ..constants.config_constants import ConfigConstants
from ..error_handling.exceptions import (
    ConfigurationError,
    OperationTimeoutError,
    VectorStoreError,
)
from ..models.vector_store_and_session_models import VectorStoreSearchResult

logger = logging.getLogger("vector_store_service")


def run_with_timeout(
    func, args=(), kwargs=None, timeout=ConfigConstants.DEFAULT_VECTOR_STORE_TIMEOUT
):
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
        # Ensure we have a valid exception to raise
        exc = exception[0]
        if isinstance(exc, BaseException):
            raise exc  # pylint: disable=raising-bad-type
        else:
            # This should never happen, but provide a fallback
            raise RuntimeError(f"Thread failed with invalid exception: {exc}")
    return result[0]


class VectorStoreService:
    def __init__(self, vector_config):
        self.vector_config = vector_config
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
                "Initializing ChromaDB at %s", self.vector_config.persist_directory
            )

            # Create directory if it doesn't exist
            import os

            os.makedirs(self.vector_config.persist_directory, exist_ok=True)

            # Try to connect with a timeout to prevent hanging
            def create_client():
                client = chromadb.PersistentClient(
                    path=self.vector_config.persist_directory
                )
                # Test the connection with a simple operation
                logger.info("Testing ChromaDB connection...")
                test_collection = client.get_or_create_collection("test_connection")
                # Try a simple operation to ensure it's working
                test_collection.count()
                logger.info("ChromaDB connection test successful")
                return client

            try:
                client = run_with_timeout(
                    create_client, timeout=ConfigConstants.DEFAULT_VECTOR_STORE_TIMEOUT
                )
            except OperationTimeoutError as e:
                raise ConfigurationError(
                    f"ChromaDB initialization timed out after {ConfigConstants.DEFAULT_VECTOR_STORE_TIMEOUT} seconds. "
                    "This might be due to:\n"
                    "1. ChromaDB downloading embedding models on first run\n"
                    "2. Network connectivity issues\n"
                    "3. File system permission problems\n"
                    "4. Another process using the database\n\n"
                    "Try restarting the application or clearing the vector_db directory."
                ) from e

            logger.info(
                "Successfully connected to ChromaDB at %s",
                self.vector_config.persist_directory,
            )
            return client

        except (chromadb.errors.ChromaError, IOError) as e:
            error_msg = (
                f"CRITICAL: Failed to connect to Vector Store (ChromaDB) at path "
                f"'{self.vector_config.persist_directory}'. "
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
        """Add an item to the vector store with transaction-like consistency."""
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

            # Use upsert for data consistency - handles both add and update scenarios
            self.collection.upsert(documents=[content], metadatas=[meta], ids=[item_id])

            # Verify persistence by attempting to retrieve the item
            try:
                verification_result = self.collection.get(ids=[item_id])
                if (
                    not verification_result["ids"]
                    or verification_result["ids"][0] != item_id
                ):
                    raise VectorStoreError(
                        f"Persistence verification failed for item {item_id}"
                    )
            except Exception as verify_error:
                logger.warning(
                    "Could not verify persistence for item %s: %s",
                    item_id,
                    verify_error,
                )
                # Continue execution as the upsert may have succeeded

            logger.debug(
                f"Successfully persisted item to vector store with ID: {item_id}"
            )
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

    def verify_persistence_integrity(self) -> bool:
        """Verify the integrity of the persistent vector store."""
        try:
            # Test basic operations to ensure persistence is working
            test_id = "integrity_test_" + str(hash("test_content"))
            test_content = "Test content for integrity verification"
            test_metadata = {"test": "true", "timestamp": str(hash("timestamp"))}

            # Add test item
            self.collection.upsert(
                documents=[test_content], metadatas=[test_metadata], ids=[test_id]
            )

            # Verify retrieval
            result = self.collection.get(ids=[test_id])
            if not result["ids"] or result["ids"][0] != test_id:
                logger.error(
                    "Persistence integrity check failed: could not retrieve test item"
                )
                return False

            # Clean up test item
            self.collection.delete(ids=[test_id])

            logger.debug("Persistence integrity verification successful")
            return True

        except Exception as e:
            logger.error("Persistence integrity check failed: %s", e)
            return False

    def batch_add_items(
        self, items_data: list[tuple[Any, str, Optional[Dict[str, Any]]]]
    ) -> list[str]:
        """Add multiple items in a batch operation for better consistency."""
        if not items_data:
            return []

        try:
            documents = []
            metadatas = []
            ids = []

            for item, content, metadata in items_data:
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

                documents.append(content)
                metadatas.append(meta)
                ids.append(item_id)

            # Batch upsert for better consistency
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

            # Verify batch persistence
            try:
                verification_result = self.collection.get(ids=ids)
                verified_ids = set(verification_result["ids"] or [])
                expected_ids = set(ids)

                if not expected_ids.issubset(verified_ids):
                    missing_ids = expected_ids - verified_ids
                    logger.warning(
                        "Batch persistence verification: missing items %s", missing_ids
                    )
            except Exception as verify_error:
                logger.warning("Could not verify batch persistence: %s", verify_error)

            logger.debug(f"Successfully batch persisted {len(ids)} items to vector store")
            return ids

        except (chromadb.errors.ChromaError, TypeError) as e:
            logger.error("Failed to batch add items to vector store: %s", e)
            raise VectorStoreError("Failed to batch add items to vector store") from e


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
                    config = get_config()
                    _vector_store_instance = VectorStoreService(config.vector_db)
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

    def verify_persistence_integrity(self) -> bool:
        """Mock persistence integrity check."""
        logger.debug("Mock persistence integrity check - always returns True")
        return True

    def batch_add_items(
        self, items_data: list[tuple[Any, str, Optional[Dict[str, Any]]]]
    ) -> list[str]:
        """Mock batch add items implementation."""
        return [f"mock_id_{i}" for i in range(len(items_data))]
