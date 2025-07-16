"""
Context Assembler

Assembles prioritized context for response generation using neural gain weighted
retrieval across RTM trees, temporal books, and vector embeddings.

This is part of the Active Workspace - the bridge between long-term memory
and short-term response generation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from ..types import RetrievalContext, RTMNode, TemporalBook, TemporalScale

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles prioritized context for the Production Module.

    Implements the Active Workspace concept:
    1. Multi-source retrieval (vectors, RTM trees, temporal books)
    2. Neural gain weighted prioritization
    3. Temporal scope filtering
    4. Context window optimization
    5. Reasoning chain tracking
    """

    def __init__(
        self,
        vector_manager=None,
        rtm_store=None,
        temporal_library=None,
        max_context_length: int = 8192
    ):
        self.vector_manager = vector_manager
        self.rtm_store = rtm_store
        self.temporal_library = temporal_library
        self.max_context_length = max_context_length

        # Context assembly strategies
        self.retrieval_strategies = {
            "hybrid": self._hybrid_retrieval,
            "vector_only": self._vector_only_retrieval,
            "temporal_only": self._temporal_retrieval,
            "rtm_traversal": self._rtm_traversal_retrieval
        }

        logger.info("Context Assembler initialized")

    async def assemble_context(
        self,
        query: str,
        max_depth: int = 3,
        temporal_scope: str | None = None,
        session_id: str | None = None,
        strategy: str = "hybrid"
    ) -> RetrievalContext:
        """
        Assemble prioritized context for a query.

        Args:
            query: User's question or request
            max_depth: Maximum RTM tree depth to traverse
            temporal_scope: Time scope filter ('hour', 'day', 'week', etc.)
            session_id: Current session for contextual relevance
            strategy: Retrieval strategy to use

        Returns:
            RetrievalContext with prioritized nodes and metadata
        """
        try:
            logger.info(f"Assembling context for query: '{query[:50]}...'")

            # Initialize context
            context = RetrievalContext(
                query=query,
                temporal_scope=TemporalScale(temporal_scope) if temporal_scope and temporal_scope != "all" else None,
                retrieval_strategy=strategy
            )

            # Step 1: Multi-source retrieval
            retrieval_func = self.retrieval_strategies.get(strategy, self._hybrid_retrieval)
            raw_results = await retrieval_func(
                query, max_depth, temporal_scope, session_id
            )

            # Step 2: Neural gain weighted prioritization
            prioritized_nodes = await self._prioritize_by_neural_gain(
                raw_results, query, context
            )

            # Step 3: Context window optimization
            optimized_nodes = await self._optimize_context_window(
                prioritized_nodes, context
            )

            # Update context with results
            context.retrieved_nodes = optimized_nodes
            context.max_salience_score = max([node.salience_score for node in optimized_nodes], default=0.0)
            context.avg_salience_score = sum([node.salience_score for node in optimized_nodes]) / len(optimized_nodes) if optimized_nodes else 0.0
            context.total_context_length = sum([len(node.content) for node in optimized_nodes])

            logger.info(f"Context assembled: {len(optimized_nodes)} nodes, avg salience: {context.avg_salience_score:.2f}")
            return context

        except Exception as e:
            logger.error(f"Error assembling context: {e}")
            # Return empty context on error
            return RetrievalContext(query=query, retrieval_strategy=strategy)

    async def _hybrid_retrieval(
        self,
        query: str,
        max_depth: int,
        temporal_scope: str | None,
        session_id: str | None
    ) -> list[RTMNode]:
        """
        Hybrid retrieval combining multiple sources.

        Strategy:
        1. Vector similarity search across RTM nodes
        2. Temporal book filtering based on scope
        3. RTM tree traversal for contextual nodes
        4. Combine and deduplicate results
        """
        results = []

        try:
            # Step 1: Vector similarity search if vector manager is available
            if self.vector_manager:
                vector_results = await self.vector_manager.query_similar_vectors(
                    query_text=query,
                    collection_name="rtm_nodes",
                    top_k=max_depth * 3,  # Get more candidates for filtering
                    salience_threshold=0.3
                )

                # Convert vector results to RTMNode objects
                for result in vector_results:
                    metadata = result.get("metadata", {})
                    node = RTMNode(
                        node_id=metadata.get("node_id", f"vector_{len(results)}"),
                        tree_id=metadata.get("tree_id", ""),
                        content=result.get("document", ""),
                        summary=metadata.get("summary", ""),
                        salience_score=result.get("salience_score", 1.0),
                        temporal_scale=TemporalScale(metadata.get("temporal_scale", "day")),
                        depth=metadata.get("depth", 0)
                    )
                    results.append(node)

            # Step 2: Temporal filtering if temporal library is available
            if self.temporal_library and temporal_scope:
                try:
                    temporal_scale = TemporalScale(temporal_scope) if temporal_scope != "all" else None
                    if temporal_scale:
                        # Get relevant temporal books
                        relevant_books: list[TemporalBook] = await self.temporal_library.find_books_by_scale(temporal_scale)

                        # Filter results to only include nodes from relevant books
                        book_tree_ids = set()
                        for book in relevant_books:
                            book_tree_ids.update(book.rtm_tree_ids)

                        results = [node for node in results if node.tree_id in book_tree_ids]
                except ValueError:
                    # Invalid temporal scope, skip temporal filtering
                    logger.warning(f"Invalid temporal scope: {temporal_scope}")

            # Step 3: RTM tree traversal for additional context if RTM store is available
            if self.rtm_store and results:
                # Get additional nodes from the same trees at the specified depth
                tree_ids = {node.tree_id for node in results if node.tree_id}

                for tree_id in list(tree_ids)[:3]:  # Limit to 3 trees for performance
                    try:
                        tree_data = await self.rtm_store.load_tree(tree_id)
                        if tree_data:
                            # Get nodes at optimal depth for context
                            context_nodes = tree_data.traverse_to_depth(max_depth)

                            # Add nodes that aren't already in results
                            existing_node_ids = {node.node_id for node in results}
                            for context_node in context_nodes:
                                if context_node.node_id not in existing_node_ids:
                                    results.append(context_node)
                    except Exception as e:
                        logger.warning(f"Error retrieving tree {tree_id}: {e}")

            # Step 4: Session-specific filtering if session_id is provided
            if session_id:
                # Prioritize nodes from the same session
                session_results = [node for node in results if getattr(node, 'session_id', '') == session_id]
                other_results = [node for node in results if getattr(node, 'session_id', '') != session_id]

                # Boost salience for session-specific nodes
                for node in session_results:
                    node.salience_score *= 1.5

                results = session_results + other_results

            logger.info(f"Hybrid retrieval found {len(results)} nodes for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            # Return empty list on error rather than mock data
            return []

    async def _vector_only_retrieval(
        self,
        query: str,
        max_depth: int,
        temporal_scope: str | None,
        session_id: str | None
    ) -> list[RTMNode]:
        """Vector-only retrieval strategy."""
        # TODO: Implement vector retrieval when vector_manager is complete
        return []

    async def _temporal_retrieval(
        self,
        query: str,
        max_depth: int,
        temporal_scope: str | None,
        session_id: str | None
    ) -> list[RTMNode]:
        """Temporal-only retrieval strategy."""
        # TODO: Implement temporal retrieval when temporal_library is complete
        return []

    async def _rtm_traversal_retrieval(
        self,
        query: str,
        max_depth: int,
        temporal_scope: str | None,
        session_id: str | None
    ) -> list[RTMNode]:
        """RTM tree traversal retrieval strategy."""
        # TODO: Implement RTM traversal when rtm_store is complete
        return []

    async def _prioritize_by_neural_gain(
        self,
        nodes: list[RTMNode],
        query: str,
        context: RetrievalContext
    ) -> list[RTMNode]:
        """
        Apply neural gain weighted prioritization.

        Uses the Inverted Encoding Model (IEM) approach where neural gain
        (salience) is encoded in the vector magnitude. Prioritizes based on:
        1. Existing salience scores (from neural gain mechanism)
        2. Query relevance boost
        3. Temporal recency
        4. Hierarchical importance
        """
        if not nodes:
            return []

        try:
            # Calculate query embedding for relevance scoring if vector manager available
            query_relevance_scores = {}
            if self.vector_manager and hasattr(self.vector_manager, 'embedding_model'):
                try:
                    query_embedding = self.vector_manager.embedding_model.encode([query])[0]

                    for node in nodes:
                        if node.base_embedding:
                            # Calculate cosine similarity for relevance boost
                            node_embedding = node.base_embedding
                            dot_product = sum(a * b for a, b in zip(query_embedding, node_embedding, strict=True))
                            norm_query = sum(x * x for x in query_embedding) ** 0.5
                            norm_node = sum(x * x for x in node_embedding) ** 0.5

                            if norm_query > 0 and norm_node > 0:
                                cosine_sim = dot_product / (norm_query * norm_node)
                                query_relevance_scores[node.node_id] = cosine_sim
                            else:
                                query_relevance_scores[node.node_id] = 0.0
                        else:
                            # Fallback: simple text similarity
                            query_lower = query.lower()
                            content_lower = (node.content + " " + node.summary).lower()
                            common_words = set(query_lower.split()) & set(content_lower.split())
                            query_relevance_scores[node.node_id] = len(common_words) / max(len(query_lower.split()), 1)
                except Exception as e:
                    logger.warning(f"Error calculating query relevance: {e}")
                    # Fallback to text-based similarity
                    for node in nodes:
                        query_lower = query.lower()
                        content_lower = (node.content + " " + node.summary).lower()
                        common_words = set(query_lower.split()) & set(content_lower.split())
                        query_relevance_scores[node.node_id] = len(common_words) / max(len(query_lower.split()), 1)
            else:
                # Fallback: simple text-based relevance
                for node in nodes:
                    query_lower = query.lower()
                    content_lower = (node.content + " " + node.summary).lower()
                    common_words = set(query_lower.split()) & set(content_lower.split())
                    query_relevance_scores[node.node_id] = len(common_words) / max(len(query_lower.split()), 1)

            # Calculate final neural gain scores
            for node in nodes:
                base_salience = node.salience_score
                query_relevance = query_relevance_scores.get(node.node_id, 0.0)

                # Temporal recency boost (more recent = higher gain)
                time_delta = (datetime.now() - node.timestamp).total_seconds() / 3600  # hours
                temporal_boost = max(0.5, 2.0 - (time_delta * 0.01))  # Decay over time

                # Hierarchical importance (root and summary nodes get boost)
                hierarchy_boost = 1.0
                if node.node_type.value in ["root", "summary"]:
                    hierarchy_boost = 1.5 - (node.depth * 0.1)
                elif node.node_type.value == "theme":
                    hierarchy_boost = 1.3

                # Calculate final neural gain
                final_gain = (
                    base_salience *
                    (1.0 + query_relevance) *
                    temporal_boost *
                    hierarchy_boost
                )

                # Update node salience score
                node.salience_score = min(final_gain, 3.0)  # Cap at maximum gain

            # Sort by neural gain (highest first)
            prioritized_nodes = sorted(nodes, key=lambda n: n.salience_score, reverse=True)

            logger.info(f"Prioritized {len(prioritized_nodes)} nodes by neural gain. Top score: {prioritized_nodes[0].salience_score:.2f}")
            return prioritized_nodes

        except Exception as e:
            logger.error(f"Error in neural gain prioritization: {e}")
            # Fallback: sort by existing salience scores
            return sorted(nodes, key=lambda n: n.salience_score, reverse=True)

    async def _optimize_context_window(
        self,
        nodes: list[RTMNode],
        context: RetrievalContext
    ) -> list[RTMNode]:
        """
        Optimize context to fit within window constraints.

        Smart optimization strategy:
        1. Prioritize high-salience nodes
        2. Ensure hierarchical diversity (don't just take leaves)
        3. Maintain narrative coherence
        4. Fit within token/character limits
        """
        if not nodes:
            return []

        try:
            # Calculate content sizes
            node_sizes = {}
            for node in nodes:
                content_size = len(node.content) + len(node.summary) + 50  # 50 chars for metadata
                node_sizes[node.node_id] = content_size

            # Strategy 1: Greedy selection based on salience-to-size ratio
            optimized = []
            total_length = 0
            used_tree_ids = set()

            # First pass: Include highest salience nodes from different trees
            for node in nodes:
                node_size = node_sizes[node.node_id]

                # Check if adding this node would exceed limit
                if total_length + node_size > self.max_context_length:
                    continue

                # Prioritize diversity across trees (don't overload from one conversation)
                if node.tree_id in used_tree_ids:
                    # Apply penalty for same-tree nodes
                    if len([n for n in optimized if n.tree_id == node.tree_id]) >= 2:
                        continue  # Skip if we already have 2+ nodes from this tree

                # Include the node
                optimized.append(node)
                total_length += node_size
                used_tree_ids.add(node.tree_id)

                # Stop if we're getting close to the limit
                if total_length > self.max_context_length * 0.9:
                    break

            # Strategy 2: Ensure hierarchical diversity
            # Make sure we have some summary/root nodes for high-level context
            hierarchy_counts = {}
            for node in optimized:
                hierarchy_counts[node.node_type.value] = hierarchy_counts.get(node.node_type.value, 0) + 1

            # If we don't have enough high-level nodes, try to add some
            if hierarchy_counts.get("root", 0) == 0 or hierarchy_counts.get("summary", 0) < 2:
                high_level_nodes = [n for n in nodes if n.node_type.value in ["root", "summary"] and n not in optimized]
                high_level_nodes.sort(key=lambda n: n.salience_score, reverse=True)

                for node in high_level_nodes[:2]:  # Add up to 2 high-level nodes
                    node_size = node_sizes[node.node_id]
                    if total_length + node_size <= self.max_context_length:
                        # Remove the lowest salience leaf node to make room if needed
                        if optimized:
                            leaf_nodes = [n for n in optimized if n.node_type.value == "leaf"]
                            if leaf_nodes:
                                least_salient_leaf = min(leaf_nodes, key=lambda n: n.salience_score)
                                if least_salient_leaf.salience_score < node.salience_score:
                                    optimized.remove(least_salient_leaf)
                                    total_length -= node_sizes[least_salient_leaf.node_id]

                        optimized.append(node)
                        total_length += node_size

            # Strategy 3: Final pass - fill remaining space with highest salience nodes
            remaining_nodes = [n for n in nodes if n not in optimized]
            remaining_nodes.sort(key=lambda n: n.salience_score, reverse=True)

            for node in remaining_nodes:
                node_size = node_sizes[node.node_id]
                if total_length + node_size <= self.max_context_length:
                    optimized.append(node)
                    total_length += node_size
                else:
                    break

            # Final sort by salience for consistent ordering
            optimized.sort(key=lambda n: n.salience_score, reverse=True)

            logger.info(f"Context optimization: {len(nodes)} -> {len(optimized)} nodes, {total_length}/{self.max_context_length} chars")
            return optimized

        except Exception as e:
            logger.error(f"Error optimizing context window: {e}")
            # Fallback: simple truncation by length
            optimized = []
            total_length = 0

            for node in nodes:
                node_length = len(node.content) + len(node.summary)
                if total_length + node_length <= self.max_context_length:
                    optimized.append(node)
                    total_length += node_length
                else:
                    break

            return optimized

    async def get_current_context(self) -> dict[str, Any]:
        """Get current active workspace context."""
        return {
            "max_context_length": self.max_context_length,
            "available_strategies": list(self.retrieval_strategies.keys()),
            "vector_manager_available": self.vector_manager is not None,
            "rtm_store_available": self.rtm_store is not None,
            "temporal_library_available": self.temporal_library is not None,
            "last_updated": datetime.now().isoformat()
        }
