"""
SVG-based Vector Manager

This module provides an enhanced vector manager that uses Support Vector Graphs (SVG)
for indexing and searching, offering formal navigability guarantees and improved
search quality, especially in high-dimensional spaces.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from ..types import (
    ConversationTurn,
    NeuralGainConfig,
    RTMNode,
    RTMTree,
    TemporalBook,
    TemporalScale,
)
from .svg_memory_index import SVGMemoryIndex

logger = logging.getLogger(__name__)

class SVGVectorManager:
    """
    An enhanced vector manager that uses SVG instead of or alongside ChromaDB.

    This can be used as a drop-in replacement for the existing vector manager
    with improved theoretical guarantees.
    """

    def __init__(
        self,
        storage_path: str,
        embedding_model: str,
        config: NeuralGainConfig | None = None,
        svg_config: dict | None = None
    ):
        """
        Initialize the SVGVectorManager.

        Args:
            storage_path (str): Path to store any related data (less relevant for in-memory SVG).
            embedding_model (str): The name of the sentence transformer model for embeddings.
            config (Optional[NeuralGainConfig]): Configuration for neural gain.
            svg_config (Optional[dict]): Configuration for the SVG index.
        """
        self.storage_path = storage_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self.config = config or NeuralGainConfig()
        svg_config = svg_config or {}

        self.svg_index = SVGMemoryIndex(
            kernel_width=svg_config.get("kernel_width", 1.0),
            max_out_degree=svg_config.get("max_out_degree", 32)
        )
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info("SVGVectorManager initialized.")

    async def store_conversation_vectors(
        self,
        conversation: list[ConversationTurn],
        rtm_tree: RTMTree,
        temporal_book: TemporalBook
    ) -> dict[str, Any]:
        """
        Index an RTM tree using SVG. This is the main entry point for indexing.
        """
        all_nodes = list(rtm_tree.nodes.values())

        for node in all_nodes:
            content_for_embedding = f"{node.summary}\n{node.content}"
            base_embedding = self.embedding_model.encode(
                content_for_embedding,
                normalize_embeddings=True
            )
            salience_score = self._calculate_node_salience(node, rtm_tree, temporal_book)
            weighted_embedding = base_embedding * salience_score

            node.base_embedding = base_embedding.tolist()
            node.salience_score = salience_score
            node.weighted_embedding = weighted_embedding.tolist()

            self.svg_index.add_node(node, weighted_embedding.cpu().numpy() if hasattr(weighted_embedding, "cpu") else np.array(weighted_embedding))

        return {
            "vectors_stored": len(all_nodes),
            "embedding_dimension": self.embedding_dim,
            "neural_gain_applied": True,
            "index_type": "SVG"
        }

    def _calculate_node_salience(
        self,
        node: RTMNode,
        rtm_tree: RTMTree,
        temporal_book: TemporalBook
    ) -> float:
        """
        Calculate salience score for neural gain weighting.
        """
        salience = self.config.base_salience

        time_delta = datetime.now() - node.timestamp
        hours_old = time_delta.total_seconds() / 3600
        temporal_decay = np.exp(-hours_old * self.config.temporal_decay_factor)
        salience *= (1 + temporal_decay)

        if node.node_type.value in ["root", "summary"]:
            depth_boost = 1.5 - (node.depth * 0.1)
            salience *= max(depth_boost, 1.0)

        content_length = len(node.content) + len(node.summary)
        if content_length > 100:
            salience *= 1.2

        scale_weights = {
            TemporalScale.MINUTE: 1.0,
            TemporalScale.HOUR: 1.1,
            TemporalScale.DAY: 1.2,
            TemporalScale.WEEK: 1.0,
            TemporalScale.MONTH: 0.9,
            TemporalScale.YEAR: 0.8
        }
        salience *= scale_weights.get(node.temporal_scale, 1.0)

        return np.clip(salience, 0.1, self.config.max_gain_multiplier)

    async def query_similar_vectors(
        self,
        query_text: str,
        top_k: int = 10,
        salience_threshold: float = 0.5,
        **kwargs
    ) -> list[dict[str, Any]]:
        """
        Search using SVG.
        """
        if not self.svg_index.graph:
            logger.warning("SVG index is not built. No search can be performed.")
            return []

        query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True)
        # Ensure query_embedding is a numpy ndarray
        if hasattr(query_embedding, "cpu"):
            query_embedding = query_embedding.cpu().numpy()
        elif hasattr(query_embedding, "numpy"):
            query_embedding = query_embedding.numpy()
        query_embedding = np.array(query_embedding)
        results = self.svg_index.search(query_embedding, k=top_k)

        processed_results = []
        for node_id, score in results:
            node = self.svg_index.nodes.get(node_id)
            if node and node.salience_score >= salience_threshold:
                processed_results.append({
                    "document": f"{node.summary} {node.content}",
                    "metadata": {
                        "id": getattr(node, "node_id", None),
                        "type": "RTMNode",
                        "salience_score": node.salience_score,
                        "timestamp": node.timestamp.isoformat()
                    },
                    "distance": 1 - score,
                    "salience_score": node.salience_score,
                    "rank": len(processed_results) + 1
                })
        return processed_results

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the SVG index."""
        stats = self.svg_index.get_statistics()
        stats["embedding_dimension"] = self.embedding_dim
        return stats

    async def close(self):
        """Clean shutdown."""
        # No explicit close needed for the in-memory graph
        logger.info("SVGVectorManager closing.")
        pass
