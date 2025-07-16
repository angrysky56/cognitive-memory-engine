"""
Core Cognitive Memory Engine

This module provides the main interface to the Cognitive Memory Engine,
integrating all components into a unified system.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..comprehension.narrative_tree_builder import NarrativeTreeBuilder
from ..comprehension.temporal_organizer import TemporalOrganizer
from ..production.response_generator import ResponseGenerator
from ..storage.rtm_graphs import RTMGraphStore
from ..storage.temporal_library import TemporalLibrary
from ..storage.vector_store import VectorStore
from ..types import ConversationTurn, SystemConfig, TemporalScale
from ..workspace.context_assembler import ContextAssembler
from ..workspace.vector_manager import VectorManager
from ..workspace.svg_vector_manager import SVGVectorManager
from ..config import get_cloud_provider_config
from .exceptions import CMEError, CMEInitializationError

logger = logging.getLogger(__name__)


class CognitiveMemoryEngine:
    """
    Main Cognitive Memory Engine class that orchestrates all memory subsystems.

    Integrates:
    - Comprehension Module (narrative trees, temporal organization)
    - Active Workspace (vector management, context assembly)
    - Production Module (response generation)
    - Storage Systems (vectors, temporal library, RTM graphs)
    """

    def __init__(self, config: SystemConfig | None = None):
        """
        Initialize the Cognitive Memory Engine.

        Args:
            config: Optional system configuration
        """
        self.config = config or SystemConfig()
        self.initialized = False

        # Core components (initialized in initialize())
        self.narrative_builder: NarrativeTreeBuilder | None = None
        self.temporal_organizer: TemporalOrganizer | None = None
        self.context_assembler: ContextAssembler | None = None
        self.vector_manager: VectorManager | None = None
        self.response_generator: ResponseGenerator | None = None
        self.vector_store: VectorStore | None = None
        self.temporal_library: TemporalLibrary | None = None
        self.rtm_store: RTMGraphStore | None = None

        # Session tracking
        self.active_sessions: dict[str, datetime] = {}

        logger.info("Cognitive Memory Engine created")

    async def initialize(self) -> None:
        """Initialize all engine components."""
        if self.initialized:
            return

        try:
            logger.info("Initializing Cognitive Memory Engine components...")

            # Ensure data directory exists
            data_dir = Path(self.config.data_directory)
            data_dir.mkdir(exist_ok=True)

            # Initialize storage systems first
            logger.info("Initializing storage systems...")

            self.vector_store = VectorStore({
                "persist_directory": str(data_dir / "chroma_db"),
                "collection_name": "cme_memory"
            })
            await self.vector_store.initialize()

            self.temporal_library = TemporalLibrary(str(data_dir / "temporal"))
            await self.temporal_library.initialize()

            self.rtm_store = RTMGraphStore(str(data_dir / "rtm_graphs"))

            # Initialize comprehension components
            logger.info("Initializing comprehension module...")
            
            # Get cloud provider configuration
            cloud_config = get_cloud_provider_config()

            self.narrative_builder = NarrativeTreeBuilder(
                cloud_config=cloud_config,
                rtm_store=self.rtm_store,
                config=self.config.rtm_config
            )

            self.temporal_organizer = TemporalOrganizer(
                temporal_library=self.temporal_library,
                vector_manager=None,  # Set after vector_manager is created
                rtm_store=self.rtm_store,  # Pass the initialized rtm_store
                config=self.config.neural_gain_config
            )

            # Initialize workspace components
            logger.info("Initializing workspace...")

            if self.config.vector_manager == 'svg':
                logger.info("Using SVGVectorManager")
                self.vector_manager = SVGVectorManager(
                    storage_path=str(data_dir / "vectors"),
                    embedding_model=self.config.embedding_model,
                    config=self.config.neural_gain_config,
                    svg_config=self.config.svg_config
                )
            else:
                logger.info("Using ChromaDB VectorManager")
                self.vector_manager = VectorManager(
                    storage_path=str(data_dir / "vectors"),
                    embedding_model=self.config.embedding_model,
                    config=self.config.neural_gain_config
                )

            # Now set vector_manager in temporal_organizer
            self.temporal_organizer.vector_manager = self.vector_manager

            self.context_assembler = ContextAssembler(
                vector_manager=self.vector_manager,
                rtm_store=self.rtm_store,
                temporal_library=self.temporal_library,
                max_context_length=self.config.max_context_length
            )

            # Initialize production components
            logger.info("Initializing production module...")

            self.response_generator = ResponseGenerator(
                cloud_config=cloud_config,
                max_response_length=1000,
                temperature=0.7
            )

            self.initialized = True
            logger.info("Cognitive Memory Engine initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Memory Engine: {e}")
            raise CMEInitializationError(f"Initialization failed: {e}") from e

    def set_model(self, model_name: str) -> None:
        """
        Sets the active LLM model for all components.

        Args:
            model_name: The name of the model to use.
        """
        if not self.initialized:
            raise CMEInitializationError("Engine must be initialized before setting a model.")

        # The provider's set_model updates the shared cloud_config object.
        if self.narrative_builder and self.narrative_builder.provider:
            self.narrative_builder.provider.set_model(model_name)
            logger.info(f"Set LLM model to '{model_name}' for all providers.")

            # Also update the convenience attributes on components
            if self.response_generator:
                self.response_generator.provider.set_model(model_name)
        else:
            logger.warning("No provider available to set model.")

    async def store_conversation(
        self,
        conversation: list[dict[str, str]],
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Store a conversation in memory with full cognitive analysis.

        Args:
            conversation: list of message dicts with 'role' and 'content'
            context: Optional metadata about the conversation

        Returns:
            dictionary with storage results and analysis
        """
        if not self.initialized:
            await self.initialize()

        try:
            timestamp = datetime.now()
            session_id = context.get("session_id", f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}")

            # Convert to ConversationTurn objects if needed
            conversation_turns = []
            for msg in conversation:
                if isinstance(msg, ConversationTurn):
                    conversation_turns.append(msg)
                else:
                    # Assume it's a dictionary
                    conversation_turns.append(ConversationTurn(
                        role=msg["role"],
                        content=msg["content"],
                        timestamp=timestamp
                    ))

            logger.info(f"Processing conversation with {len(conversation_turns)} turns")

            # Step 1: Build narrative tree using RTM algorithm
            rtm_tree = await self.narrative_builder.build_tree_from_conversation(
                conversation_turns,
                session_id,
                TemporalScale.DAY
            )

            # Step 1a: Store the RTM tree
            if self.rtm_store:
                await self.rtm_store.store_tree(rtm_tree)

            logger.info(f"Built and stored RTM tree {rtm_tree.tree_id} with {rtm_tree.node_count} nodes")

            # Step 2: Organize temporally into books and shelves
            temporal_book = await self.temporal_organizer.organize_conversation(
                conversation_turns,
                rtm_tree,
                session_id
            )

            logger.info(f"Organized into temporal book {temporal_book.book_id}")

            # Step 3: Store vectors with neural gain weighting
            vector_results = await self.vector_manager.store_conversation_vectors(
                conversation_turns,
                rtm_tree,
                temporal_book
            )

            logger.info("Stored vectors with neural gain weighting")

            # Track session
            self.active_sessions[session_id] = timestamp

            # Safely extract serializable data
            result = {
                "conversation_id": str(rtm_tree.tree_id),
                "session_id": str(session_id),
                "timestamp": timestamp.isoformat(),
                "message_count": len(conversation),
                "rtm_tree": {
                    "tree_id": str(rtm_tree.tree_id),
                    "node_count": getattr(rtm_tree, 'node_count', 0),
                    "compression_ratio": getattr(rtm_tree, 'compression_ratio', 0.0),
                    "max_depth": getattr(rtm_tree, 'max_recall_depth', 0)
                },
                "temporal_book": {
                    "book_id": str(temporal_book.book_id),
                    "temporal_scale": str(temporal_book.temporal_scale.value) if hasattr(temporal_book.temporal_scale, 'value') else str(temporal_book.temporal_scale),
                    "shelf_category": str(temporal_book.shelf_category.value) if hasattr(temporal_book.shelf_category, 'value') else str(temporal_book.shelf_category),
                    "persistent_themes": list(temporal_book.persistent_themes) if temporal_book.persistent_themes else []
                },
                "vector_storage": vector_results,
                "status": "success"
            }

            logger.info(f"Successfully stored conversation {rtm_tree.tree_id}")
            return result

        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            raise CMEError(f"Failed to store conversation: {e}") from e

    async def query_memory(
        self,
        query: str,
        context_depth: int = 3,
        time_scope: str = "week",
        max_results: int = 10
    ) -> dict[str, Any]:
        """
        Search memory using semantic and temporal constraints.

        Args:
            query: Natural language query
            context_depth: RTM tree traversal depth
            time_scope: Temporal scope ('hour', 'day', 'week', 'month', 'year', 'all')
            max_results: Maximum results to return

        Returns:
            dictionary with search results and context
        """
        if not self.initialized:
            await self.initialize()

        try:
            logger.info(f"Querying memory: '{query}' (depth: {context_depth}, scope: {time_scope})")

            # Step 1: Find relevant temporal books using themes and time scope
            temporal_scope_enum = None
            if time_scope != "all":
                temporal_scope_map = {
                    "hour": TemporalScale.HOUR,
                    "day": TemporalScale.DAY,
                    "week": TemporalScale.WEEK,
                    "month": TemporalScale.MONTH,
                    "year": TemporalScale.YEAR
                }
                temporal_scope_enum = temporal_scope_map.get(time_scope)

            relevant_books = await self.temporal_organizer.get_relevant_books(
                query,
                temporal_scope_enum,
                max_books=max_results
            )

            logger.info(f"Found {len(relevant_books)} relevant temporal books")

            # Step 2: Assemble context using hybrid retrieval
            context = await self.context_assembler.assemble_context(
                query=query,
                max_depth=context_depth,
                temporal_scope=time_scope,
                session_id=None,  # Cross-session query
                strategy="hybrid"
            )

            # Step 3: Format results
            results = []
            for book in relevant_books:
                # Get RTM trees from book
                for tree_id in book.rtm_tree_ids[:3]:  # Limit per book
                    results.append({
                        "type": "temporal_book",
                        "book_id": book.book_id,
                        "tree_id": tree_id,
                        "title": book.title,
                        "themes": book.persistent_themes,
                        "timestamp": book.last_accessed.isoformat(),
                        "temporal_scale": book.temporal_scale.value,
                        "relevance_score": 0.8  # TODO: Implement proper scoring
                    })

            result = {
                "query": query,
                "context_depth": context_depth,
                "time_scope": time_scope,
                "total_books_searched": len(relevant_books),
                "results": results[:max_results],
                "assembled_context": {
                    "retrieved_nodes": len(context.retrieved_nodes),
                    "temporal_books_accessed": len(context.temporal_books_accessed),
                    "avg_salience_score": context.avg_salience_score,
                    "retrieval_strategy": context.retrieval_strategy
                },
                "search_timestamp": datetime.now().isoformat()
            }

            logger.info(f"Query completed: found {len(results)} results")
            return result

        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            raise CMEError(f"Memory query failed: {e}") from e

    async def generate_response(
        self,
        prompt: str,
        context_depth: int = 3,
        response_style: str = "conversational"
    ) -> str:
        """
        Generate a contextually aware response using memory.

        Args:
            prompt: Input prompt
            context_depth: Memory context depth
            response_style: Style of response

        Returns:
            Generated response string
        """
        if not self.initialized:
            await self.initialize()

        try:
            logger.info(f"Generating response for: '{prompt[:50]}...'")

            # Step 1: Query relevant memory context
            await self.query_memory(
                prompt, context_depth, "week", 5
            )

            # Step 2: Assemble context for response generation
            context = await self.context_assembler.assemble_context(
                query=prompt,
                max_depth=context_depth,
                temporal_scope="week",
                strategy="hybrid"
            )

            # Step 3: Generate response with memory context
            response = await self.response_generator.generate_response(
                query=prompt,
                context=context,
                include_social_context=True,
                response_type=response_style
            )

            logger.info(f"Generated response ({response.confidence:.2f} confidence)")
            return response.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise CMEError(f"Response generation failed: {e}") from e

    async def analyze_conversation(
        self,
        conversation_id: str | None = None,
        analysis_type: str = "all"
    ) -> dict[str, Any]:
        """
        Perform deep analysis of conversation patterns.

        Args:
            conversation_id: Specific conversation to analyze (None for recent)
            analysis_type: Type of analysis ('narrative', 'temporal', 'semantic', 'social', 'all')

        Returns:
            Analysis results dictionary
        """
        if not self.initialized:
            await self.initialize()

        try:
            # For now, return analysis of recent conversations if no ID specified
            if conversation_id is None:
                recent_conversations = await self.get_recent_conversations(1)
                if not recent_conversations:
                    raise CMEError("No conversations found for analysis")
                conversation_data = recent_conversations[0]
                conversation_id = conversation_data.get("id", "unknown")

            analysis = {
                "conversation_id": conversation_id,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }

            if analysis_type in ["narrative", "all"]:
                # TODO: Implement narrative analysis
                analysis["narrative"] = {
                    "structure": "hierarchical_rtm",
                    "depth": 3,
                    "themes": ["project", "timeline", "technical"]
                }

            if analysis_type in ["temporal", "all"]:
                # Get temporal organization stats
                stats = await self.temporal_organizer.get_statistics()
                analysis["temporal"] = stats

            if analysis_type in ["semantic", "all"]:
                # TODO: Implement semantic analysis
                analysis["semantic"] = {
                    "embedding_dimensions": 384,
                    "semantic_clusters": 5,
                    "topic_coherence": 0.75
                }

            logger.info(f"Completed {analysis_type} analysis for {conversation_id}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing conversation: {e}")
            raise CMEError(f"Conversation analysis failed: {e}") from e

    async def get_memory_stats(self, include_details: bool = False) -> dict[str, Any]:
        """
        Get comprehensive memory system statistics.

        Args:
            include_details: Include detailed component breakdowns

        Returns:
            Statistics dictionary
        """
        if not self.initialized:
            await self.initialize()

        try:
            stats = {
                "engine_status": "initialized",
                "timestamp": datetime.now().isoformat(),
                "active_sessions": len(self.active_sessions)
            }

            # Vector store stats
            if self.vector_store:
                vector_stats = await self.vector_store.get_stats()
                stats["vector_store"] = vector_stats

            # Temporal library stats
            if self.temporal_library:
                temporal_stats = await self.temporal_organizer.get_statistics()
                stats["temporal_library"] = temporal_stats

            if include_details and self.narrative_builder:
                # Component details
                stats["components"] = {
                    "narrative_builder": {
                        "llm_model": self.narrative_builder.llm_model,
                        "rtm_config": {
                            "max_branching_factor": self.narrative_builder.config.max_branching_factor,
                            "max_recall_depth": self.narrative_builder.config.max_recall_depth
                        }
                    },
                    "response_generator": {
                        "llm_model": self.response_generator.llm_model,
                        "generation_stats": self.response_generator.generation_stats
                    }
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            raise CMEError(f"Failed to get memory stats: {e}") from e

    async def get_recent_conversations(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent conversations from temporal library."""
        if not self.initialized:
            await self.initialize()

        if not self.temporal_library:
            return []

        # Get recent books and extract conversation info
        try:
            books = await self.temporal_library.find_books_by_scale(TemporalScale.DAY)

            # Sort by last accessed and take most recent
            books.sort(key=lambda b: b.last_accessed, reverse=True)

            conversations = []
            for book in books[:limit]:
                for tree_id in book.rtm_tree_ids:
                    conversations.append({
                        "id": tree_id,
                        "book_id": book.book_id,
                        "title": book.title,
                        "timestamp": book.last_accessed.isoformat(),
                        "themes": book.persistent_themes,
                        "temporal_scale": book.temporal_scale.value
                    })

                    if len(conversations) >= limit:
                        break

                if len(conversations) >= limit:
                    break

            return conversations

        except Exception as e:
            logger.error(f"Error getting recent conversations: {e}")
            return []

    async def get_narrative_summaries(self) -> list[dict[str, Any]]:
        """Get narrative tree summaries."""
        if not self.initialized or not self.rtm_store:
            return []

        return await self.rtm_store.list_trees(limit=20)

    async def get_temporal_organization(self) -> dict[str, Any]:
        """Get temporal organization structure."""
        if not self.initialized:
            await self.initialize()

        if not self.temporal_organizer:
            return {}

        return await self.temporal_organizer.get_statistics()

    async def get_active_context(self) -> dict[str, Any]:
        """Get current active workspace context."""
        if not self.initialized or not self.context_assembler:
            return {}

        # Assemble context for a generic, recent query
        context = await self.context_assembler.assemble_context(
            query="What is the current status?",
            max_depth=2,
            temporal_scope="day",
            strategy="hybrid"
        )

        return {
            "active_sessions": len(self.active_sessions),
            "last_updated": datetime.now().isoformat(),
            "context_assembly_ready": True,
            "retrieved_nodes": len(context.retrieved_nodes),
            "avg_salience_score": context.avg_salience_score
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up Cognitive Memory Engine")

        try:
            if self.vector_store:
                await self.vector_store.cleanup()
            if self.temporal_library:
                await self.temporal_library.cleanup()
            if self.rtm_store:
                await self.rtm_store.close()

            self.active_sessions.clear()
            self.initialized = False
            logger.info("Cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
