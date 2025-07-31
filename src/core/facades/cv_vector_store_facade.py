"""Facade for managing vector store operations."""

from typing import Any, Dict, List, Optional

from src.config.logging_config import get_structured_logger
from src.constants.config_constants import ConfigConstants
from src.error_handling.exceptions import VectorStoreError
from src.models.workflow_models import ContentType
from src.services.vector_store_service import VectorStoreService


class CVVectorStoreFacade:
    """Provides a simplified interface for vector store operations."""

    def __init__(self, vector_db: Optional[VectorStoreService]):
        self._vector_db = vector_db
        self.logger = get_structured_logger(__name__)

    async def store_content(
        self,
        content: str,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store content in vector database."""
        if not self._vector_db:
            self.logger.warning(
                "Vector database not initialized. Cannot store content."
            )
            return None

        try:
            item_id = self._vector_db.add_item(
                item=content,
                content=content,
                metadata={"content_type": content_type.value, **(metadata or {})},
            )
            return item_id
        except (VectorStoreError, TypeError, ValueError) as e:
            self.logger.error(
                "Failed to store content",
                extra={"content_type": content_type.value, "error": str(e)},
            )
            return None

    async def search_content(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        limit: int = ConfigConstants.DEFAULT_SEARCH_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Search for similar content."""
        if not self._vector_db:
            self.logger.warning(
                "Vector database not initialized. Cannot search content."
            )
            return []

        try:
            results = self._vector_db.search(query=query, k=limit)
            return results
        except (VectorStoreError, TypeError, ValueError) as e:
            self.logger.error(
                "Failed to search content",
                extra={
                    "query": query,
                    "content_type": content_type.value if content_type else None,
                    "error": str(e),
                },
            )
            return []

    async def find_similar_content(
        self,
        content: str,
        content_type: Optional[ContentType] = None,
        limit: int = ConfigConstants.DEFAULT_SIMILAR_CONTENT_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Find content similar to the provided content."""
        if not self._vector_db:
            self.logger.warning(
                "Vector database not initialized. Cannot find similar content."
            )
            return []

        try:
            results = self._vector_db.search(query=content, k=limit)
            return results
        except (VectorStoreError, TypeError, ValueError) as e:
            self.logger.error(
                "Failed to find similar content",
                extra={
                    "content_type": content_type.value if content_type else None,
                    "error": str(e),
                },
            )
            return []
