"""Vector store configuration interface for dependency injection.

This module defines the interface that vector store services need,
following the Interface Segregation Principle to avoid contract breaches.
"""

from typing import Protocol


class VectorStoreConfigInterface(Protocol):
    """Interface for vector store configuration.

    This protocol defines the minimal configuration interface that
    vector store services require, following the Interface Segregation
    Principle to avoid passing entire configuration objects.
    """

    @property
    def persist_directory(self) -> str:
        """Get the persist directory for the vector store."""

    @property
    def collection_name(self) -> str:
        """Get the collection name for the vector store."""

    @property
    def embedding_model(self) -> str:
        """Get the embedding model name."""

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""

    @property
    def max_search_results(self) -> int:
        """Get the maximum search results."""

    @property
    def similarity_threshold(self) -> float:
        """Get the similarity threshold."""
