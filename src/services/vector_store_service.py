import chromadb
from src.utils.exceptions import ConfigurationError
from src.config.settings import get_config
import logging

logger = logging.getLogger("vector_store_service")


class VectorStoreService:
    def __init__(self, settings=None):
        self.settings = settings or get_config()
        self.client = self._connect()

    def _connect(self):
        try:
            client = chromadb.PersistentClient(path=self.settings.vector_db.persist_directory)
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

    def get_client(self):
        return self.client


_vector_store_instance = None


def get_vector_store_service():
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
    return _vector_store_instance
