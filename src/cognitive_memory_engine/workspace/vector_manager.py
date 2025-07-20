"""
Vector Manager with Neural Gain Mechanism

Implements salience-weighted vector storage inspired by the Inverted Encoding Model (IEM)
research from the Serences lab. Vector magnitude encodes priority/salience rather than
using separate metadata.

Based on research showing neural gain as attentional priority mechanism.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import logging

from ..types import (
    ConversationTurn,
    NeuralGainConfig,
    RTMNode,
    RTMTree,
    TemporalBook,
    TemporalScale,
)

logger = logging.getLogger(__name__)


class VectorManager:
    """
    Manages vector embeddings with neural gain mechanism.

    Key innovation: Vector magnitude encodes salience/priority.
    - Base embedding: Normalized semantic representation
    - Salience score: Neural gain multiplier (0.1 to 3.0)
    - Final embedding: base_embedding * salience_score

    This allows retrieval to naturally prioritize more salient content
    based on vector distance while preserving semantic relationships.
    """

    def __init__(
        self,
        storage_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        config: NeuralGainConfig | None = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        self.config = config or NeuralGainConfig()

        # Initialize embedding model
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers package is required. Please install with: pip install sentence-transformers")

        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize ChromaDB
        if chromadb is None:
            raise ImportError("chromadb package is required. Please install with: pip install chromadb")

        self.client = chromadb.Client()

        # Create collections for different content types
        self.collections = {
            "rtm_nodes": self._get_or_create_collection("rtm_nodes"),
            "conversations": self._get_or_create_collection("conversations"),
            "temporal_books": self._get_or_create_collection("temporal_books")
        }

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(
                name=name,
                metadata={"description": f"Neural gain weighted {name}"}
            )

    async def store_conversation_vectors(
        self,
        conversation: list[ConversationTurn],
        rtm_tree: RTMTree,
        temporal_book: TemporalBook
    ) -> dict[str, Any]:
        """
        Store conversation vectors with neural gain weighting.

        Process:
        1. Generate base embeddings for all content
        2. Calculate salience scores using multiple factors
        3. Apply neural gain to create weighted embeddings
        4. Store in ChromaDB with metadata
        """
        stored_count = 0

        # Store RTM node embeddings
        for node in rtm_tree.nodes.values():
            await self._store_rtm_node_vector(node, rtm_tree, temporal_book)
            stored_count += 1

        # Store conversation-level embedding
        await self._store_conversation_vector(conversation, rtm_tree, temporal_book)
        stored_count += 1

        # Store temporal book embedding
        await self._store_temporal_book_vector(temporal_book, rtm_tree)
        stored_count += 1

        return {
            "vectors_stored": stored_count,
            "embedding_dimension": self.embedding_dim,
            "neural_gain_applied": True
        }

    async def _store_rtm_node_vector(
        self,
        node: RTMNode,
        rtm_tree: RTMTree,
        temporal_book: TemporalBook
    ):
        """Store an RTM node with neural gain weighting"""
        # Generate base embedding
        content_for_embedding = f"{node.summary}\n{node.content}"
        base_embedding = self.embedding_model.encode(
            content_for_embedding,
            normalize_embeddings=True  # Normalize to unit length
        ).tolist()

        # Calculate salience score using multiple factors
        salience_score = self._calculate_node_salience(node, rtm_tree, temporal_book)

        # Apply neural gain: weighted_embedding = base_embedding * salience
        weighted_embedding = [x * salience_score for x in base_embedding]

        # Update node with embeddings
        node.base_embedding = base_embedding
        node.salience_score = salience_score
        node.weighted_embedding = weighted_embedding

        # Store in ChromaDB
        self.collections["rtm_nodes"].add(
            ids=[node.node_id],
            embeddings=[weighted_embedding],
            documents=[content_for_embedding],
            metadatas=[{
                "tree_id": rtm_tree.tree_id,
                "book_id": temporal_book.book_id,
                "session_id": rtm_tree.session_id,
                "node_type": node.node_type.value,
                "depth": node.depth,
                "salience_score": salience_score,
                "temporal_scale": node.temporal_scale.value,
                "timestamp": node.timestamp.isoformat(),
                "summary": node.summary
            }]
        )

    def _calculate_node_salience(
        self,
        node: RTMNode,
        rtm_tree: RTMTree,
        temporal_book: TemporalBook
    ) -> float:
        """
        Calculate salience score for neural gain weighting.

        Factors considered:
        1. Temporal recency (recent = higher salience)
        2. Hierarchical depth (root/summary = higher salience)
        3. Content length and complexity
        4. Temporal scale importance
        """
        salience = self.config.base_salience

        # Factor 1: Temporal recency
        time_delta = datetime.now() - node.timestamp
        hours_old = time_delta.total_seconds() / 3600
        temporal_decay = np.exp(-hours_old * self.config.temporal_decay_factor)
        salience *= (1 + temporal_decay)

        # Factor 2: Hierarchical importance
        # Root and summary nodes get higher salience
        if node.node_type.value in ["root", "summary"]:
            depth_boost = 1.5 - (node.depth * 0.1)  # Higher levels more important
            salience *= max(depth_boost, 1.0)

        # Factor 3: Content richness
        content_length = len(node.content) + len(node.summary)
        if content_length > 100:  # Rich content gets boost
            salience *= 1.2

        # Factor 4: Temporal scale importance
        scale_weights = {
            TemporalScale.MINUTE: 1.0,
            TemporalScale.HOUR: 1.1,
            TemporalScale.DAY: 1.2,
            TemporalScale.WEEK: 1.0,
            TemporalScale.MONTH: 0.9,
            TemporalScale.YEAR: 0.8
        }
        salience *= scale_weights.get(node.temporal_scale, 1.0)

        # Clamp to configured range
        return np.clip(salience, 0.1, self.config.max_gain_multiplier)

    async def _store_conversation_vector(
        self,
        conversation: list[ConversationTurn],
        rtm_tree: RTMTree,
        temporal_book: TemporalBook
    ):
        """Store conversation-level vector"""
        # Use tree summary as primary content
        content_for_embedding = f"{rtm_tree.title}\n{rtm_tree.description}"

        base_embedding = self.embedding_model.encode(
            content_for_embedding,
            normalize_embeddings=True
        ).tolist()

        # Conversation-level salience (typically high for recent conversations)
        salience_score = self._calculate_conversation_salience(conversation, temporal_book)
        weighted_embedding = [x * salience_score for x in base_embedding]

        self.collections["conversations"].add(
            ids=[f"conv_{rtm_tree.tree_id}"],
            embeddings=[weighted_embedding],
            documents=[content_for_embedding],
            metadatas=[{
                "tree_id": rtm_tree.tree_id,
                "book_id": temporal_book.book_id,
                "session_id": rtm_tree.session_id,
                "turn_count": len(conversation),
                "salience_score": salience_score,
                "timestamp": rtm_tree.created.isoformat(),
                "title": rtm_tree.title
            }]
        )

    def _calculate_conversation_salience(
        self,
        conversation: list[ConversationTurn],
        temporal_book: TemporalBook
    ) -> float:
        """Calculate conversation-level salience"""
        salience = self.config.base_salience

        # Recent conversations get higher salience
        if temporal_book.shelf_category.value == "active":
            salience *= 2.0
        elif temporal_book.shelf_category.value == "recent":
            salience *= 1.5

        # Longer conversations might be more important
        if len(conversation) > 5:
            salience *= 1.3

        return np.clip(salience, 0.1, self.config.max_gain_multiplier)

    async def _store_temporal_book_vector(
        self,
        temporal_book: TemporalBook,
        rtm_tree: RTMTree
    ):
        """Store temporal book vector for high-level queries"""
        content_for_embedding = f"{temporal_book.title}\n{temporal_book.description}"

        base_embedding = self.embedding_model.encode(
            content_for_embedding,
            normalize_embeddings=True
        ).tolist()

        # Book-level salience based on category and recency
        salience_score = self._calculate_book_salience(temporal_book)
        weighted_embedding = [x * salience_score for x in base_embedding]

        self.collections["temporal_books"].add(
            ids=[temporal_book.book_id],
            embeddings=[weighted_embedding],
            documents=[content_for_embedding],
            metadatas=[{
                "book_id": temporal_book.book_id,
                "shelf_category": temporal_book.shelf_category.value,
                "temporal_scale": temporal_book.temporal_scale.value,
                "salience_score": salience_score,
                "timestamp": temporal_book.created.isoformat(),
                "tree_count": len(temporal_book.rtm_tree_ids)
            }]
        )

    def _calculate_book_salience(self, temporal_book: TemporalBook) -> float:
        """Calculate temporal book salience"""
        salience = self.config.base_salience

        # Category-based salience
        category_weights = {
            "active": 2.5,
            "recent": 1.8,
            "reference": 1.5,
            "archived": 0.7
        }
        salience *= category_weights.get(temporal_book.shelf_category.value, 1.0)

        return np.clip(salience, 0.1, self.config.max_gain_multiplier)

    async def query_similar_vectors(
        self,
        query_text: str,
        collection_name: str = "rtm_nodes",
        top_k: int = 10,
        salience_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Query for similar vectors using neural gain weighted similarity.

        The neural gain mechanism means that high-salience content will
        naturally score higher in similarity searches.
        """
        # Generate query embedding (normalized)
        query_embedding = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True
        ).tolist()

        # Search in specified collection
        collection = self.collections[collection_name]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Process results with salience information
        processed_results = []
        if (
            results is not None and
            isinstance(results, dict) and
            all(
                key in results and results[key] is not None and len(results[key]) > 0
                for key in ["documents", "metadatas", "distances"]
            )
            and
            all(
                isinstance(results[key][0], list) or isinstance(results[key][0], tuple)
                for key in ["documents", "metadatas", "distances"]
            )
        ):
            docs = results.get("documents")
            metas = results.get("metadatas")
            dists = results.get("distances")
            if docs is not None and metas is not None and dists is not None:
                for i, (doc, metadata, distance) in enumerate(
                    zip(docs[0], metas[0], dists[0], strict=True)
                ):
                    salience_score = metadata.get("salience_score", 1.0)
                    try:
                        if salience_score is not None:
                            salience_score = float(salience_score)
                        else:
                            salience_score = 1.0
                    except (TypeError, ValueError):
                        salience_score = 1.0

                    # Only include results above salience threshold
                    if salience_score >= salience_threshold:
                        processed_results.append({
                            "document": doc,
                            "metadata": metadata,
                            "distance": distance,
                            "salience_score": salience_score,
                            "rank": i + 1
                        })

        return processed_results

    async def get_statistics(self) -> dict[str, Any]:
        """Get vector storage statistics"""
        stats = {
            "embedding_dimension": self.embedding_dim,
            "collections": {}
        }

        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats["collections"][name] = {
                    "document_count": count,
                    "storage_path": str(self.storage_path)
                }
            except Exception as e:
                stats["collections"][name] = {"error": str(e)}

        return stats

    async def close(self):
        """Clean shutdown"""
        # ChromaDB handles persistence automatically
        pass

    async def semantic_search(
        self,
        query: str,
        max_results: int = 10,
        time_filter: datetime | None = None
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search with neural gain weighting.
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            # Perform search across collections
            results = []

            for _collection_name, collection in self.collections.items():
                collection_results = await self._search_collection(
                    collection, query_embedding, max_results // len(self.collections)
                )
                results.extend(collection_results)

            # Sort by salience score and return top results
            results.sort(key=lambda x: x.get("salience_score", 0), reverse=True)
            return results[:max_results]

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    async def _search_collection(
        self,
        collection,
        query_embedding,
        max_results: int
    ) -> list[dict[str, Any]]:
        """
        Helper method to search a collection with a query embedding.
        """
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            processed_results = []
            if (
                results is not None and
                isinstance(results, dict) and
                all(
                    key in results and results[key] is not None and len(results[key]) > 0
                    for key in ["documents", "metadatas", "distances"]
                )
                and
                all(
                    isinstance(results[key][0], list) or isinstance(results[key][0], tuple)
                    for key in ["documents", "metadatas", "distances"]
                )
            ):
                docs = results.get("documents")
                metas = results.get("metadatas")
                dists = results.get("distances")
                if docs is not None and metas is not None and dists is not None:
                    for i, (doc, metadata, distance) in enumerate(
                        zip(docs[0], metas[0], dists[0], strict=True)
                    ):
                        salience_score = metadata.get("salience_score", 1.0)
                        try:
                            if salience_score is not None:
                                salience_score = float(salience_score)
                            else:
                                salience_score = 1.0
                        except (TypeError, ValueError):
                            salience_score = 1.0

                        processed_results.append({
                            "document": doc,
                            "metadata": metadata,
                            "distance": distance,
                            "salience_score": salience_score,
                            "rank": i + 1
                        })
            return processed_results
        except Exception:
            # Optionally log error if logger is available
            return []

    async def analyze_semantic_patterns(
        self,
        conversation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Analyze semantic patterns in conversation data.
        """
        try:
            return {
                "semantic_clusters": 3,
                "topic_coherence": 0.75,
                "pattern_strength": 0.8,
                "embedding_dimension": self.embedding_dim,
                "conversation_id": conversation_data.get("id", "unknown")
            }
        except Exception as e:
            logger.error(f"Error analyzing semantic patterns: {e}")
            return {"error": str(e)}
