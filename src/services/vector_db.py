from typing import List, Dict, Any, Callable, Optional, Tuple
import faiss
import numpy as np
import google.generativeai as genai
from src.core.state_manager import VectorStoreConfig, SkillEntry, ExperienceEntry
from uuid import uuid4
import logging
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib

from ..models.data_models import ContentType, ProcessingStatus
from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..services.error_recovery import get_error_recovery_service

# Set up logging
logger = get_structured_logger("vector_db")


@dataclass
class EnhancedVectorDocument:
    """Enhanced document for vector database with Phase 1 integration."""
    id: str
    content: str
    content_type: ContentType
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with additional metadata."""
    document: EnhancedVectorDocument
    similarity_score: float
    rank: int
    relevance_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.relevance_metadata is None:
            self.relevance_metadata = {}


class VectorDB:
    """
    Enhanced vector database for skills and experiences with Phase 1 infrastructure.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        embed_function: Optional[Callable[[str], List[float]]] = None,
        db_path: Optional[str] = None
    ):
        """
        Initializes the VectorDB with the given configuration.

        Args:
            config: The VectorStoreConfig object.
            embed_function: Optional. A function to generate embeddings.
                             If None, uses a default Gemini embedding function.
            db_path: Optional path for persistent storage.
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

        # Enhanced features
        self.settings = get_config()
        self.error_recovery = get_error_recovery_service()
        self.db_path = Path(db_path or self.settings.data_directory) / "enhanced_vector_db"
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Enhanced documents storage
        self.enhanced_documents: Dict[str, EnhancedVectorDocument] = {}

        # Performance tracking
        self.stats = {
            "documents_stored": 0,
            "searches_performed": 0,
            "average_search_time": 0.0,
            "embedding_cache_hits": 0,
            "last_updated": datetime.now()
        }

        # Embedding cache for performance
        self.embedding_cache: Dict[str, List[float]] = {}

        # Load existing enhanced data
        self._load_enhanced_database()

        logger.info(
            "Enhanced VectorDB initialized",
            config_dimension=config.dimension,
            index_type=config.index_type,
            enhanced_documents=len(self.enhanced_documents),
            legacy_documents=len(self.data)
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

    # Enhanced methods for Phase 1 integration

    def _load_enhanced_database(self):
        """Load enhanced database from disk."""
        try:
            db_file = self.db_path / "enhanced_documents.pkl"
            if db_file.exists():
                with open(db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.enhanced_documents = data.get('documents', {})
                    self.stats = data.get('stats', self.stats)
                    self.embedding_cache = data.get('embedding_cache', {})

                logger.info(
                    "Enhanced vector database loaded",
                    document_count=len(self.enhanced_documents)
                )
        except Exception as e:
            logger.error("Failed to load enhanced vector database", error=str(e))
            self.enhanced_documents = {}

    def _save_enhanced_database(self):
        """Save enhanced database to disk."""
        try:
            db_file = self.db_path / "enhanced_documents.pkl"
            data = {
                'documents': self.enhanced_documents,
                'stats': self.stats,
                'embedding_cache': self.embedding_cache
            }

            with open(db_file, 'wb') as f:
                pickle.dump(data, f)

            logger.debug("Enhanced vector database saved")

        except Exception as e:
            logger.error("Failed to save enhanced vector database", error=str(e))

    def _generate_document_id(self, content: str, content_type: ContentType) -> str:
        """Generate unique document ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{content_type.value}_{content_hash}_{timestamp}"

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            self.stats["embedding_cache_hits"] += 1
            return self.embedding_cache[text_hash]
        return None

    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding for future use."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.embedding_cache[text_hash] = embedding

        # Limit cache size
        if len(self.embedding_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.embedding_cache.keys())[:100]
            for key in oldest_keys:
                del self.embedding_cache[key]

    async def add_enhanced_document(
        self,
        content: str,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Add an enhanced document to the vector database."""
        try:
            # Generate ID if not provided
            if document_id is None:
                document_id = self._generate_document_id(content, content_type)

            # Check cache first
            embedding = self._get_cached_embedding(content)
            if embedding is None:
                # Generate embedding with error recovery
                try:
                    embedding = await self.error_recovery.execute_with_retry(
                        lambda: self._embed_function(content),
                        max_retries=3,
                        operation_name="generate_embedding"
                    )
                    self._cache_embedding(content, embedding)
                except Exception as e:
                    logger.error(
                        "Failed to generate embedding",
                        error=str(e),
                        content_length=len(content)
                    )
                    # Use zero vector as fallback
                    embedding = [0.0] * self.config.dimension

            # Create enhanced document
            document = EnhancedVectorDocument(
                id=document_id,
                content=content,
                content_type=content_type,
                embedding=embedding,
                metadata=metadata or {},
                session_id=session_id
            )

            # Store document
            self.enhanced_documents[document_id] = document

            # Update stats
            self.stats["documents_stored"] = len(self.enhanced_documents)
            self.stats["last_updated"] = datetime.now()

            # Save to disk
            self._save_enhanced_database()

            logger.info(
                "Enhanced document added",
                document_id=document_id,
                content_type=content_type.value,
                content_length=len(content),
                session_id=session_id
            )

            return document_id

        except Exception as e:
            logger.error(
                "Failed to add enhanced document",
                error=str(e),
                content_type=content_type.value
            )
            raise

    async def search_enhanced(
        self,
        query: str,
        content_type: Optional[ContentType] = None,
        limit: int = 10,
        min_similarity: float = 0.1,
        session_id: Optional[str] = None
    ) -> List[EnhancedSearchResult]:
        """Enhanced search with better similarity calculation."""
        start_time = datetime.now()

        try:
            # Generate query embedding with caching
            query_embedding = self._get_cached_embedding(query)
            if query_embedding is None:
                query_embedding = await self.error_recovery.execute_with_retry(
                    lambda: self._embed_function(query),
                    max_retries=2,
                    operation_name="generate_query_embedding"
                )
                self._cache_embedding(query, query_embedding)

            # Calculate similarities
            results = []

            for doc_id, document in self.enhanced_documents.items():
                # Filter by content type if specified
                if content_type and document.content_type != content_type:
                    continue

                # Filter by session if specified
                if session_id and document.session_id != session_id:
                    continue

                # Calculate similarity
                if document.embedding:
                    similarity = self._calculate_cosine_similarity(
                        query_embedding, document.embedding
                    )

                    if similarity >= min_similarity:
                        results.append((document, similarity))

            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)

            # Limit results
            results = results[:limit]

            # Create enhanced search results
            search_results = [
                EnhancedSearchResult(
                    document=doc,
                    similarity_score=sim,
                    rank=idx + 1,
                    relevance_metadata={
                        "query_length": len(query),
                        "content_length": len(doc.content),
                        "search_timestamp": datetime.now().isoformat()
                    }
                )
                for idx, (doc, sim) in enumerate(results)
            ]

            # Update stats
            search_time = (datetime.now() - start_time).total_seconds()
            self.stats["searches_performed"] += 1

            # Update average search time
            current_avg = self.stats["average_search_time"]
            search_count = self.stats["searches_performed"]
            self.stats["average_search_time"] = (
                (current_avg * (search_count - 1) + search_time) / search_count
            )

            logger.info(
                "Enhanced search completed",
                query_length=len(query),
                results_count=len(search_results),
                search_time=search_time,
                content_type=content_type.value if content_type else "all",
                session_id=session_id
            )

            return search_results

        except Exception as e:
            logger.error(
                "Enhanced search failed",
                error=str(e),
                query=query[:100]
            )
            return []

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)

        except Exception as e:
            logger.error("Similarity calculation failed", error=str(e))
            return 0.0

    async def get_enhanced_document(self, document_id: str) -> Optional[EnhancedVectorDocument]:
        """Get an enhanced document by ID."""
        return self.enhanced_documents.get(document_id)

    async def update_enhanced_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an enhanced document."""
        try:
            document = self.enhanced_documents.get(document_id)
            if not document:
                logger.warning("Enhanced document not found for update", document_id=document_id)
                return False

            # Update content and regenerate embedding if needed
            if content is not None:
                document.content = content
                # Check cache first
                embedding = self._get_cached_embedding(content)
                if embedding is None:
                    embedding = await self.error_recovery.execute_with_retry(
                        lambda: self._embed_function(content),
                        max_retries=2,
                        operation_name="update_embedding"
                    )
                    self._cache_embedding(content, embedding)
                document.embedding = embedding

            # Update metadata
            if metadata is not None:
                document.metadata.update(metadata)

            # Update timestamp
            document.updated_at = datetime.now()

            # Save to disk
            self._save_enhanced_database()

            logger.info("Enhanced document updated", document_id=document_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to update enhanced document",
                error=str(e),
                document_id=document_id
            )
            return False

    async def delete_enhanced_document(self, document_id: str) -> bool:
        """Delete an enhanced document."""
        try:
            if document_id in self.enhanced_documents:
                del self.enhanced_documents[document_id]

                # Update stats
                self.stats["documents_stored"] = len(self.enhanced_documents)
                self.stats["last_updated"] = datetime.now()

                # Save to disk
                self._save_enhanced_database()

                logger.info("Enhanced document deleted", document_id=document_id)
                return True
            else:
                logger.warning("Enhanced document not found for deletion", document_id=document_id)
                return False

        except Exception as e:
            logger.error(
                "Failed to delete enhanced document",
                error=str(e),
                document_id=document_id
            )
            return False

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced database statistics."""
        return {
            **self.stats,
            "documents_by_type": self._get_enhanced_documents_by_type(),
            "cache_size": len(self.embedding_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "database_size_mb": self._get_enhanced_database_size()
        }

    def _get_enhanced_documents_by_type(self) -> Dict[str, int]:
        """Get enhanced document count by content type."""
        type_counts = {}
        for document in self.enhanced_documents.values():
            content_type = document.content_type.value
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        return type_counts

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate embedding cache hit rate."""
        total_requests = self.stats["searches_performed"] + self.stats["documents_stored"]
        if total_requests == 0:
            return 0.0
        return self.stats["embedding_cache_hits"] / total_requests

    def _get_enhanced_database_size(self) -> float:
        """Get enhanced database size in MB."""
        try:
            db_file = self.db_path / "enhanced_documents.pkl"
            if db_file.exists():
                size_bytes = db_file.stat().st_size
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except Exception:
            return 0.0

    async def cleanup_old_enhanced_documents(self, days_old: int = 30) -> int:
        """Clean up enhanced documents older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0

            documents_to_delete = [
                doc_id for doc_id, doc in self.enhanced_documents.items()
                if doc.created_at < cutoff_date
            ]

            for doc_id in documents_to_delete:
                await self.delete_enhanced_document(doc_id)
                deleted_count += 1

            logger.info(
                "Old enhanced documents cleaned up",
                deleted_count=deleted_count,
                cutoff_days=days_old
            )

            return deleted_count

        except Exception as e:
            logger.error("Failed to cleanup old enhanced documents", error=str(e))
            return 0


# Global enhanced vector database instance
_enhanced_vector_db = None


def get_enhanced_vector_db(config: Optional[VectorStoreConfig] = None) -> VectorDB:
    """Get the global enhanced vector database instance."""
    global _enhanced_vector_db
    if _enhanced_vector_db is None:
        if config is None:
            # Create default config
            config = VectorStoreConfig(
                dimension=768,  # Standard embedding dimension
                index_type="IndexFlatL2"
            )
        _enhanced_vector_db = VectorDB(config)
    return _enhanced_vector_db


# Convenience functions for enhanced functionality
async def store_enhanced_content(
    content: str,
    content_type: ContentType,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> str:
    """Store content in enhanced vector database."""
    db = get_enhanced_vector_db()
    return await db.add_enhanced_document(content, content_type, metadata, session_id=session_id)


async def search_enhanced_content(
    query: str,
    content_type: Optional[ContentType] = None,
    limit: int = 5,
    session_id: Optional[str] = None
) -> List[EnhancedSearchResult]:
    """Search for similar content in enhanced database."""
    db = get_enhanced_vector_db()
    return await db.search_enhanced(query, content_type, limit, session_id=session_id)


async def find_enhanced_content_examples(
    content_type: ContentType,
    limit: int = 10,
    session_id: Optional[str] = None
) -> List[EnhancedVectorDocument]:
    """Find example content of a specific type from enhanced database."""
    db = get_enhanced_vector_db()
    results = await db.search_enhanced(
        query="professional experience",  # Generic query
        content_type=content_type,
        limit=limit,
        min_similarity=0.0,  # Get all documents of this type
        session_id=session_id
    )
    return [result.document for result in results]
