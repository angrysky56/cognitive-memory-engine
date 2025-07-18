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
        enhanced_knowledge_tools=None,  # Add enhanced knowledge tools for fetch data
        max_context_length: int = 8192
    ):
        self.vector_manager = vector_manager
        self.rtm_store = rtm_store
        self.temporal_library = temporal_library
        self.enhanced_knowledge_tools = enhanced_knowledge_tools  # For real-time fetch
        self.max_context_length = max_context_length

        # Context assembly strategies
        self.retrieval_strategies = {
            "hybrid": self._hybrid_retrieval,
            "hybrid_with_fetch": self._hybrid_retrieval_with_fetch,  # New strategy
            "vector_only": self._vector_only_retrieval,
            "temporal_only": self._temporal_retrieval,
            "rtm_traversal": self._rtm_traversal_retrieval,
            "fetch_enhanced": self._fetch_enhanced_retrieval  # Pure fetch strategy
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

    async def _hybrid_retrieval_with_fetch(
        self,
        query: str,
        max_depth: int,
        temporal_scope: str | None,
        session_id: str | None
    ) -> list[RTMNode]:
        """
        Hybrid retrieval that includes real-time fetched content.

        This strategy:
        1. Performs standard hybrid retrieval from stored knowledge
        2. Identifies if fresh information might be needed
        3. Fetches relevant URLs/searches in real-time
        4. Converts fetched content to RTMNode format for AI processing
        5. Combines and prioritizes all sources
        """
        results = []

        try:
            # Step 1: Get stored knowledge using standard hybrid retrieval
            stored_results = await self._hybrid_retrieval(query, max_depth, temporal_scope, session_id)
            results.extend(stored_results)

            # Step 2: Determine if fresh content is needed
            if self.enhanced_knowledge_tools and self._should_fetch_fresh_content(query, stored_results):
                logger.info(f"Fetching fresh content for query: {query[:50]}...")

                # Step 3: Fetch fresh content (no AI processing - direct ingestion)
                fresh_content = await self._fetch_fresh_content_for_query(query, max_depth)

                # Step 4: Convert fetched content to RTMNode format
                fetch_nodes = self._convert_fetch_to_rtm_nodes(fresh_content, query)

                # Step 5: Boost salience for fresh content (it's current/relevant)
                for node in fetch_nodes:
                    node.salience_score *= 1.3  # Fresh content gets priority boost

                results.extend(fetch_nodes)

                logger.info(f"Added {len(fetch_nodes)} fresh content nodes")

            return results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval with fetch: {e}")
            # Fallback to standard hybrid retrieval
            return await self._hybrid_retrieval(query, max_depth, temporal_scope, session_id)

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

    async def _fetch_enhanced_retrieval(
        self,
        query: str,
        max_depth: int,
        temporal_scope: str | None,
        session_id: str | None
    ) -> list[RTMNode]:
        """
        Pure fetch-enhanced retrieval for real-time information needs.

        This strategy prioritizes fresh content over stored knowledge.
        Ideal for current events, recent developments, or when stored
        knowledge appears stale or insufficient.
        """
        results = []

        try:
            if not self.enhanced_knowledge_tools:
                logger.warning("Enhanced knowledge tools not available for fetch retrieval")
                # Fallback to hybrid retrieval
                return await self._hybrid_retrieval(query, max_depth, temporal_scope, session_id)

            # Step 1: Fetch fresh content aggressively
            fresh_content = await self._fetch_fresh_content_for_query(query, max_depth * 2)

            # Step 2: Convert to RTMNode format
            fetch_nodes = self._convert_fetch_to_rtm_nodes(fresh_content, query)

            # Step 3: Get minimal stored context for background
            stored_results = await self._hybrid_retrieval(query, max_depth // 2, temporal_scope, session_id)

            # Step 4: Prioritize fresh over stored
            for node in fetch_nodes:
                node.salience_score *= 1.5  # Strong boost for fresh content

            for node in stored_results:
                node.salience_score *= 0.7  # Reduce stored content priority

            results = fetch_nodes + stored_results

            logger.info(f"Fetch-enhanced retrieval: {len(fetch_nodes)} fresh + {len(stored_results)} stored nodes")
            return results

        except Exception as e:
            logger.error(f"Error in fetch-enhanced retrieval: {e}")
            # Fallback to standard hybrid retrieval
            return await self._hybrid_retrieval(query, max_depth, temporal_scope, session_id)

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
            "enhanced_knowledge_tools_available": self.enhanced_knowledge_tools is not None,
            "fetch_capabilities": {
                "real_time_content": self.enhanced_knowledge_tools is not None,
                "url_extraction": True,
                "search_integration": self.enhanced_knowledge_tools is not None,
                "fresh_content_detection": True
            },
            "last_updated": datetime.now().isoformat()
        }

    def _should_fetch_fresh_content(self, query: str, stored_results: list[RTMNode]) -> bool:
        """
        Determine if fresh content should be fetched based on query analysis.

        Fetch fresh content when:
        1. Query contains time-sensitive keywords
        2. Stored results are sparse or low-quality
        3. Query asks for recent information
        4. No relevant stored content found
        """
        # Time-sensitive keywords that suggest fresh content is needed
        time_sensitive_keywords = {
            'recent', 'latest', 'current', 'new', 'today', 'yesterday',
            'this week', 'this month', 'now', 'updated', 'breaking',
            'fresh', 'live', 'real-time', '2024', '2025'
        }

        query_lower = query.lower()

        # Check for time-sensitive keywords
        if any(keyword in query_lower for keyword in time_sensitive_keywords):
            return True

        # Check if stored results are insufficient
        if len(stored_results) < 2:
            return True

        # Check if stored results have low average salience
        if stored_results:
            avg_salience = sum(node.salience_score for node in stored_results) / len(stored_results)
            if avg_salience < 0.5:
                return True

        # Check if stored content is old (basic heuristic)
        if stored_results:
            now = datetime.now()
            recent_threshold = now - timedelta(days=7)
            recent_count = sum(1 for node in stored_results if node.timestamp > recent_threshold)
            if recent_count / len(stored_results) < 0.3:  # Less than 30% recent content
                return True

        return False

    async def _fetch_fresh_content_for_query(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """
        Fetch fresh content relevant to the query using enhanced knowledge tools.

        Returns raw fetch results for conversion to RTMNode format.
        """
        fresh_content = []

        try:
            if not self.enhanced_knowledge_tools:
                return fresh_content

            # Strategy 1: Search for current information
            if hasattr(self.enhanced_knowledge_tools, '_perform_search'):
                search_results = await self.enhanced_knowledge_tools._perform_search(query, max_results)
                for result in search_results:
                    fresh_content.append({
                        'type': 'search_result',
                        'content': result.get('content', ''),
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'source': 'web_search'
                    })

            # Strategy 2: If query looks like it needs specific URL content
            urls_in_query = self._extract_urls_from_query(query)
            for url in urls_in_query:
                if hasattr(self.enhanced_knowledge_tools, '_fetch_content_from_url'):
                    url_content = await self.enhanced_knowledge_tools._fetch_content_from_url(url)
                    fresh_content.append({
                        'type': 'url_fetch',
                        'content': url_content,
                        'title': f'Content from {url}',
                        'url': url,
                        'source': 'direct_fetch'
                    })

            return fresh_content

        except Exception as e:
            logger.error(f"Error fetching fresh content: {e}")
            return fresh_content

    def _convert_fetch_to_rtm_nodes(self, fetch_results: list[dict[str, Any]], query: str) -> list[RTMNode]:
        """
        Convert raw fetch results to RTMNode format for AI processing.

        This creates temporary RTMNodes that can be processed alongside
        stored knowledge without requiring permanent storage.
        """
        nodes = []

        try:
            for i, result in enumerate(fetch_results):
                # Create a temporary RTMNode from fetch result
                node = RTMNode(
                    node_id=f"fetch_{i}_{hash(result.get('url', query)) % 10000}",
                    tree_id="fresh_content",  # Special tree ID for fetched content
                    content=result.get('content', ''),
                    summary=result.get('title', '')[:200] + "..." if len(result.get('title', '')) > 200 else result.get('title', ''),
                    salience_score=1.0,  # Will be adjusted by calling method
                    temporal_scale=TemporalScale("hour"),  # Fresh content is immediate
                    depth=0,  # Fetched content is at surface level
                    timestamp=datetime.now(),  # Mark as fresh
                    metadata={
                        'source': result.get('source', 'unknown'),
                        'url': result.get('url', ''),
                        'fetch_type': result.get('type', 'unknown'),
                        'is_fresh_content': True,
                        'query_context': query[:100]  # Store query context
                    }
                )

                nodes.append(node)

        except Exception as e:
            logger.error(f"Error converting fetch results to RTMNodes: {e}")

        return nodes

    def _extract_urls_from_query(self, query: str) -> list[str]:
        """Extract URLs from user query for direct fetching."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, query)
        return urls[:3]  # Limit to 3 URLs for performance
