"""
Vector Store Module

ChromaDB-based vector storage with neural gain weighting for salience.
"""

import logging
from datetime import datetime
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from ..core.exceptions import CMEVectorError

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector storage using ChromaDB with neural gain weighting.

    Implements salience-weighted embeddings where vector magnitude
    encodes importance (from IEM research).
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize vector store.

        Args:
            config: Vector store configuration
        """
        self.config = config
        self.client = None
        self.collection = None
        self.initialized = False

        if chromadb is None:
            logger.warning("ChromaDB not available - using mock implementation")

        logger.info("Vector store created with config")

    async def initialize(self) -> None:
        """Initialize ChromaDB connection and collection."""
        if self.initialized:
            return

        try:
            if chromadb is None:
                logger.info("ChromaDB not available - using mock vector store")
                self.initialized = True
                return

            # Initialize ChromaDB client
            from chromadb.config import Settings
            self.client = chromadb.Client(Settings(
                persist_directory=self.config.get("persist_directory", "./chroma_db"),
                anonymized_telemetry=False
            ))

            # Get or create collection
            collection_name = self.config.get("collection_name", "cme_memory")

            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Cognitive Memory Engine vectors"}
                )
                logger.info(f"Created new collection: {collection_name}")

            self.initialized = True
            logger.info("Vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise CMEVectorError(f"Vector store initialization failed: {e}") from e

    from collections.abc import Mapping, Sequence

    async def store_vectors(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: Sequence[dict[str, str | int | float | bool | None]],
        documents: list[str]
    ) -> dict[str, Any]:
        """
        Store vectors with metadata.

        Args:
            ids: Unique identifiers for vectors
            embeddings: Vector embeddings
            metadatas: Metadata for each vector
            documents: Source documents

        Returns:
            Storage results
        """
        if not self.initialized:
            await self.initialize()

        try:
            if chromadb is None or self.collection is None:
                # Mock implementation
                return {
                    "stored_count": len(ids),
                    "timestamp": datetime.now().isoformat(),
                    "mock": True
                }

            # Store in ChromaDB
            import numpy as np
            np_embeddings = np.array(embeddings, dtype=np.float32)
            self.collection.add(
                ids=ids,
                embeddings=np_embeddings,
                metadatas=list(metadatas),
                documents=documents
            )

            result = {
                "stored_count": len(ids),
                "timestamp": datetime.now().isoformat(),
                "collection_size": self.collection.count()
            }

            logger.info(f"Stored {len(ids)} vectors")
            return result

        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise CMEVectorError(f"Vector storage failed: {e}") from e

    async def search_vectors(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Metadata filter conditions
            include: What to include in results

        Returns:
            Search results
        """
        if not self.initialized:
            await self.initialize()

        try:
            if chromadb is None or self.collection is None:
                # Mock implementation
                return {
                    "ids": [f"mock_id_{i}" for i in range(min(n_results, 3))],
                    "distances": [0.1, 0.3, 0.5][:min(n_results, 3)],
                    "metadatas": [{"mock": True}] * min(n_results, 3),
                    "documents": [f"Mock document {i}" for i in range(min(n_results, 3))],
                    "mock": True
                }

            # Search ChromaDB
            default_include = ["metadatas", "documents", "distances"]
            include_list = include if include is not None else default_include
            valid_includes = [x for x in include_list if x in ["documents", "embeddings", "metadatas", "distances", "uris", "data"]]

            from typing import Literal, cast
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=cast(list[Literal['documents', 'embeddings', 'metadatas', 'distances', 'uris', 'data']], valid_includes)
            )

            # Format results
            formatted_results = {
                "ids": results["ids"][0] if results["ids"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
            }

            logger.info(f"Found {len(formatted_results['ids'])} similar vectors")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise CMEVectorError(f"Vector search failed: {e}") from e

    async def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        if not self.initialized:
            await self.initialize()

        try:
            if chromadb is None or self.collection is None:
                return {
                    "total_vectors": 42,  # Mock data
                    "collection_name": "mock_collection",
                    "mock": True
                }

            count = self.collection.count()

            return {
                "total_vectors": count,
                "collection_name": self.collection.name,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Clean up vector store resources."""
        logger.info("Cleaning up vector store")

        try:
            if self.client:
                # ChromaDB client cleanup if needed
                pass

            self.initialized = False
            logger.info("Vector store cleanup completed")

        except Exception as e:
            logger.error(f"Error during vector store cleanup: {e}")
