"""
Core Cognitive Memory Engine

This module provides the main interface to the Cognitive Memory Engine,
integrating all components into a unified system.

DUAL-TRACK ARCHITECTURE IMPLEMENTATION
=====================================

The CME implements a three-track memory architecture:

Track 1: Conversation Memory (Implemented)
- store_conversation() - Stores dialogues as narrative RTM trees
- Temporal organization by session/day
- Compression using Random Tree Model

Track 2: Document Knowledge (Implemented)
- store_document_knowledge() - Stores formal documents as concept hierarchies
- DocumentRTM with structured knowledge organization
- Root concept approach with hierarchical breakdown

Track 3: Blended Integration (Implemented)
- link_conversation_to_knowledge() - Creates cross-references
- query_blended_knowledge() - Unified retrieval interface
- Cross-reference conversation mentions to formal concepts

Storage Architecture:
- Conversation trees in rtm_graphs/
- Document knowledge in document_knowledge/
- Temporal organization in temporal/
- Vector embeddings with neural gain modulation

Query Interfaces:
- query_memory() - Search conversation memory
- get_concept() - Direct concept retrieval
- browse_knowledge_shelf() - Domain-based browsing
- query_blended_knowledge() - Unified search across both tracks
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognitive_memory_engine.mcp_server.enhanced_server_tools import (
        EnhancedKnowledgeServerTools,
    )

from ..comprehension.document_knowledge_builder import DocumentKnowledgeBuilder
from ..comprehension.narrative_tree_builder import NarrativeTreeBuilder
from ..comprehension.temporal_organizer import TemporalOrganizer
from ..list_models import get_cloud_provider_config
from ..production.response_generator import ResponseGenerator
from ..storage.cross_reference_store import CrossReferenceStore
from ..storage.document_store import DocumentStore
from ..storage.rtm_graphs import RTMGraphStore
from ..storage.temporal_library import TemporalLibrary
from ..storage.vector_store import VectorStore
from ..types import (
    ConversationTurn,
    KnowledgeDomain,
    LinkRelationship,
    NeuralGainConfig,
    RTMConfig,
    SystemConfig,
    TemporalScale,
)
from ..workspace.context_assembler import ContextAssembler
from ..workspace.svg_vector_manager import SVGVectorManager
from ..workspace.vector_manager import VectorManager
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
        self.config = config or SystemConfig(
            data_directory="data",
            llm_model="gpt-4.1-nano",
            embedding_model="all-MiniLM-L6-v2",
            rtm_config=RTMConfig(
                max_branching_factor=4,
                max_recall_depth=6,
                max_summary_length=150
            ),
            neural_gain_config=NeuralGainConfig(
                base_salience=1.0,
                temporal_decay_factor=0.1,
                max_gain_multiplier=3.0
            ),
            max_context_length=4096,
            vector_similarity_threshold=0.8,
            auto_archive_days=30
        )

        # Initialize all components directly in the constructor

        data_dir = Path(self.config.data_directory)
        data_dir.mkdir(exist_ok=True)

        # Initialize storage systems
        self.vector_store = VectorStore({
            "persist_directory": str(data_dir / "chroma_db"),
            "collection_name": "cme_memory"
        })
        # Note: In a real async application, we'd need an async constructor
        # or a factory pattern. For now, we assume sync init is sufficient
        # for object creation, and async operations happen in methods.

        self.temporal_library = TemporalLibrary(str(data_dir / "temporal"))
        self.rtm_store = RTMGraphStore(str(data_dir / "rtm_graphs"))
        self.document_store = DocumentStore(str(data_dir / "document_knowledge"))
        self.cross_reference_store = CrossReferenceStore(str(data_dir / "cross_references"))

        # Get cloud provider configuration
        cloud_config = get_cloud_provider_config()

        # Initialize comprehension components
        self.document_builder = DocumentKnowledgeBuilder(
            cloud_config=cloud_config,
            document_store=self.document_store
        )
        self.narrative_builder = NarrativeTreeBuilder(
            cloud_config=cloud_config,
            rtm_store=self.rtm_store,
            config=self.config.rtm_config
        )

        # Initialize workspace components
        if self.config.vector_manager == 'svg':
            self.vector_manager = SVGVectorManager(
                storage_path=str(data_dir / "vectors"),
                embedding_model=self.config.embedding_model,
                config=self.config.neural_gain_config,
                svg_config=self.config.svg_config
            )
        else:
            self.vector_manager = VectorManager(
                storage_path=str(data_dir / "vectors"),
                embedding_model=self.config.embedding_model,
                config=self.config.neural_gain_config
            )

        self.temporal_organizer = TemporalOrganizer(
            temporal_library=self.temporal_library,
            vector_manager=self.vector_manager,
            rtm_store=self.rtm_store,
            config=self.config.neural_gain_config
        )

        self.context_assembler = ContextAssembler(
            vector_manager=self.vector_manager,
            rtm_store=self.rtm_store,
            temporal_library=self.temporal_library,
            max_context_length=self.config.max_context_length
        )

        # Initialize production components
        self.response_generator = ResponseGenerator(
            cloud_config=cloud_config,
            max_response_length=1000,
            temperature=0.7
        )

        self.initialized = True
        self.enhanced_knowledge_tools: EnhancedKnowledgeServerTools | None = None
        self.active_sessions: dict[str, datetime] = {}
        logger.info("Cognitive Memory Engine created and components initialized")

    def _ensure_initialized(self):
        """Ensure the engine is initialized before use."""
        if not self.initialized:
            # This should not happen with the new structure, but kept for safety
            raise CMEError("Engine is not initialized.")

    def _ensure_component(self, component, name: str):
        """Ensure a component is available, raise error if not."""
        if component is None:
            raise CMEError(f"{name} component not available. Engine may not be properly initialized.")

    async def initialize(self) -> None:
        """
        Asynchronously initialize components that require it (like database connections).
        """
        if not self.initialized:
             # Fallback if somehow __init__ was bypassed or failed silently
            self.__init__(self.config)

        logger.info("Running async initializations...")
        await self.vector_store.initialize()
        await self.temporal_library.initialize()
        logger.info("Async initializations complete.")

    @property
    def llm_provider(self):
        """Get the LLM provider from the narrative builder."""
        if self.narrative_builder and self.narrative_builder.provider:
            return self.narrative_builder.provider
        elif self.response_generator and self.response_generator.provider:
            return self.response_generator.provider
        return None

    def get_current_model(self) -> str:
        """Get the currently selected model."""
        provider = self.llm_provider
        if provider:
            return provider.get_current_model()
        raise CMEError("No LLM provider available.")

    async def get_available_models(self) -> list[str]:
        """Get available models from the provider."""
        provider = self.llm_provider
        if provider:
            return await provider.get_available_models()
        raise CMEError("No LLM provider available.")

    def set_model(self, model_name: str) -> None:
        """
        Sets the active LLM model for all components.

        Args:
            model_name: The name of the model to use.
        """
        if not self.initialized:
            raise CMEInitializationError("Engine must be initialized before setting a model.")

        # The provider's set_model updates the shared cloud_config object.
        provider = self.llm_provider
        if provider:
            provider.set_model(model_name)
            logger.info(f"Set LLM model to '{model_name}' for all providers.")
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

        self._ensure_component(self.narrative_builder, "narrative_builder")
        self._ensure_component(self.temporal_organizer, "temporal_organizer")
        self._ensure_component(self.vector_manager, "vector_manager")

        try:
            timestamp = datetime.now()
            context = context or {}  # Ensure context is not None
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
            if self.rtm_store and rtm_tree:
                await self.rtm_store.store_tree(rtm_tree)

            if rtm_tree:
                logger.info(f"Built and stored RTM tree {rtm_tree.tree_id} with {rtm_tree.node_count} nodes")

            # Step 2: Organize temporally into books and shelves
            if not self.temporal_organizer:
                raise CMEError("Temporal organizer not initialized")
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

    # DUAL-TRACK ARCHITECTURE METHODS
    # ================================

    async def store_document_knowledge(
        self,
        document_content: str,
        root_concept: str,
        domain: str,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Store formal document as structured knowledge RTM.

        This creates:
        - Root node for main concept (e.g., "SAPE")
        - Hierarchical breakdown of document structure
        - Concept nodes for each major component
        - Cross-linkable knowledge graph (not conversation narrative)

        Args:
            document_content: Raw document text
            root_concept: Main concept name (e.g., "SAPE")
            domain: Knowledge domain string (converted to enum)
            metadata: Optional source metadata

        Returns:
            Storage results with document ID and analysis
        """
        if not self.initialized:
            await self.initialize()

        self._ensure_component(self.document_builder, "document_builder")
        self._ensure_component(self.document_store, "document_store")

        try:
            # Convert domain string to enum
            try:
                domain_enum = KnowledgeDomain(domain.lower())
            except ValueError:
                logger.warning(f"Unknown domain '{domain}', using GENERAL_KNOWLEDGE")
                domain_enum = KnowledgeDomain.GENERAL_KNOWLEDGE

            logger.info(f"Storing document knowledge for '{root_concept}' in domain {domain_enum.value}")

            # Use document builder to create and store DocumentRTM
            document_id = await self.document_builder.store_document_knowledge(
                document_content=document_content,
                root_concept=root_concept,
                domain=domain_enum,
                metadata=metadata
            )

            # Get the stored document for analysis
            document = await self.document_store.get_document(document_id)

            if not document:
                raise CMEError("Document was stored but could not be retrieved")

            # Create result summary
            result = {
                "document_id": document_id,
                "root_concept": root_concept,
                "domain": domain_enum.value,
                "storage_type": "formal_knowledge",
                "timestamp": datetime.now().isoformat(),
                "document_analysis": {
                    "total_concepts": document.total_concepts,
                    "compression_ratio": document.compression_ratio,
                    "max_depth": document.max_depth,
                    "root_concept_id": document.root_concept_id
                },
                "concept_hierarchy": {
                    concept_id: {
                        "name": concept.name,
                        "description": concept.description,
                        "children_count": len(concept.child_concept_ids),
                        "salience_score": concept.salience_score
                    }
                    for concept_id, concept in document.concepts.items()
                },
                "metadata": document.source_metadata,
                "status": "success"
            }

            logger.info(f"Successfully stored document knowledge '{root_concept}' with {document.total_concepts} concepts")
            return result

        except Exception as e:
            logger.error(f"Failed to store document knowledge for '{root_concept}': {e}")
            raise CMEError(f"Document knowledge storage failed: {e}") from e

    async def link_conversation_to_knowledge(
        self,
        conversation_id: str,
        document_concept_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Create cross-references between conversation and document knowledge.

        Analyzes conversation for mentions of formal concepts and creates
        bidirectional links for blended retrieval using semantic similarity.

        Args:
            conversation_id: ID of conversation to analyze
            document_concept_id: Optional specific concept to link to

        Returns:
            List of concept links created
        """
        if not self.initialized:
            await self.initialize()

        self._ensure_component(self.document_store, "document_store")
        self._ensure_component(self.cross_reference_store, "cross_reference_store")

        logger.info(f"Creating cross-references for conversation {conversation_id}")

        try:
            # Import semantic similarity calculator
            from ..semantic.similarity_calculator import SemanticSimilarityCalculator
            similarity_calc = SemanticSimilarityCalculator(self.config.embedding_model)

            # Get all document concepts for matching
            all_documents = await self.document_store.get_all_documents()
            all_concepts = []

            for doc in all_documents:
                for concept_id, concept in doc.concepts.items():
                    # Filter by specific concept if requested
                    if document_concept_id and concept_id != document_concept_id:
                        continue
                    all_concepts.append((concept_id, concept))

            # Load the conversation RTM tree
            conversation_tree = await self.rtm_store.load_tree(conversation_id)
            if not conversation_tree:
                logger.warning(f"Conversation {conversation_id} not found")
                return []

            # Analyze conversation nodes for concept mentions using semantic similarity
            created_links = []

            for node in conversation_tree.nodes.values():
                if not node or not hasattr(node, 'content') or not hasattr(node, 'node_id'):
                    continue

                # Skip very short content
                if len(node.content) < 10:
                    continue

                # Find semantically similar concepts
                similar_concepts = similarity_calc.find_similar_concepts(
                    node.content,
                    all_concepts,
                    threshold=0.4  # Minimum similarity threshold
                )

                # Create links for similar concepts
                for concept, similarity_score in similar_concepts[:3]:  # Top 3 matches per node
                    # Determine relationship type based on similarity and content
                    relationship_type = LinkRelationship.DISCUSSES
                    if similarity_score > 0.8:
                        relationship_type = LinkRelationship.ELABORATES
                    elif "?" in node.content:
                        relationship_type = LinkRelationship.QUESTIONS

                    # Create ConceptLink object
                    from ..types import ConceptLink
                    link = ConceptLink(
                        conversation_node_id=node.node_id,
                        conversation_tree_id=conversation_id,
                        document_concept_id=concept.concept_id,
                        document_id=concept.concept_id.split('_')[0] if '_' in concept.concept_id else '',
                        relationship_type=relationship_type,
                        confidence_score=similarity_score,
                        context_snippet=node.content[:150] + "..." if len(node.content) > 150 else node.content,
                        metadata={
                            'concept_name': concept.name,
                            'node_type': node.node_type.value,
                            'node_depth': node.depth
                        }
                    )

                    # Persist the link
                    if await self.cross_reference_store.store_link(link):
                        # Convert to dict for response
                        created_links.append({
                            'link_id': link.link_id,
                            'conversation_node_id': link.conversation_node_id,
                            'document_concept_id': link.document_concept_id,
                            'document_id': link.document_id,
                            'relationship_type': link.relationship_type.value,
                            'confidence_score': link.confidence_score,
                            'concept_name': concept.name,
                            'context_snippet': link.context_snippet
                        })

            logger.info(f"Created and persisted {len(created_links)} cross-reference links")
            return created_links

        except Exception as e:
            logger.error(f"Error creating cross-references: {e}")
            raise CMEError(f"Cross-reference creation failed: {e}") from e
            return []

    async def query_blended_knowledge(
        self,
        query: str,
        include_formal: bool = True,
        include_conversational: bool = True
    ) -> dict[str, Any]:
        """
        Unified query interface combining both knowledge tracks.

        Uses semantic similarity for better matching across both
        conversation and document knowledge.

        Returns:
        - formal_knowledge: Structured info from document RTMs
        - conversation_insights: Context from dialogue RTMs
        - cross_references: Links between the tracks
        - unified_summary: Blended understanding

        Args:
            query: Natural language query
            include_formal: Include formal document knowledge
            include_conversational: Include conversation insights

        Returns:
            BlendedQueryResult with combined knowledge
        """
        if not self.initialized:
            await self.initialize()

        self._ensure_component(self.document_store, "document_store")
        self._ensure_component(self.cross_reference_store, "cross_reference_store")

        logger.info(f"Blended query: '{query[:50]}...'")

        result = {
            'query': query,
            'formal_knowledge': [],
            'conversation_insights': {},
            'cross_references': [],
            'unified_summary': '',
            'confidence_score': 0.0
        }

        try:
            # Import semantic similarity calculator
            from ..semantic.similarity_calculator import SemanticSimilarityCalculator
            similarity_calc = SemanticSimilarityCalculator(self.config.embedding_model)
            # Track 1: Query formal document knowledge with semantic similarity
            if include_formal:
                all_documents = await self.document_store.get_all_documents()
                formal_matches = []

                # Collect all concepts with their documents
                all_concepts = []
                for doc in all_documents:
                    for _concept_id, concept in doc.concepts.items():
                        all_concepts.append((concept, doc))

                # Find semantically similar concepts
                for concept, doc in all_concepts:
                    # Calculate semantic similarity
                    concept_text = f"{concept.name} {concept.description} {concept.content}"
                    similarity_score = similarity_calc.calculate_similarity(query, concept_text)

                    if similarity_score > 0.3:  # Threshold for relevance
                        formal_matches.append({
                            'concept_id': concept.concept_id,
                            'document_id': doc.doc_id,
                            'concept': concept,
                            'relevance_score': similarity_score,
                            'document_title': doc.root_concept,
                            'concept_name': concept.name,
                            'description': concept.description[:200] + "..." if len(concept.description) > 200 else concept.description
                        })

                # Sort by relevance and take top matches
                formal_matches.sort(key=lambda x: x['relevance_score'], reverse=True)
                result['formal_knowledge'] = formal_matches[:5]

            # Track 2: Query conversation memory
            if include_conversational:
                conversation_results = await self.query_memory(
                    query=query,
                    context_depth=3,
                    max_results=5
                )

                result['conversation_insights'] = {
                    'results': conversation_results.get('results', []),
                    'context_summary': conversation_results.get('context_summary', ''),
                    'total_results': conversation_results.get('total_results', 0)
                }

            # Track 3: Retrieve persisted cross-references
            if result['formal_knowledge'] or result['conversation_insights']['results']:
                # Get all cross-references with high confidence
                all_links = await self.cross_reference_store.get_all_links(min_confidence=0.5)

                # Filter links relevant to our query results
                relevant_links = []

                # Check formal knowledge matches
                for match in result['formal_knowledge']:
                    concept_id = match['concept_id']
                    # Find links for this concept
                    concept_links = [link for link in all_links if link.document_concept_id == concept_id]

                    for link in concept_links[:2]:  # Top 2 links per concept
                        relevant_links.append({
                            'link_id': link.link_id,
                            'formal_concept': match['concept_name'],
                            'conversation_fragment': link.context_snippet,
                            'relationship': link.relationship_type.value,
                            'confidence': link.confidence_score,
                            'conversation_id': link.conversation_tree_id
                        })

                # Sort by confidence and limit
                relevant_links.sort(key=lambda x: x['confidence'], reverse=True)
                result['cross_references'] = relevant_links[:10]

            # Generate unified summary
            summary_parts = []

            if result['formal_knowledge']:
                formal_summary = f"Formal knowledge: Found {len(result['formal_knowledge'])} relevant concepts"
                if result['formal_knowledge']:
                    top_concept = result['formal_knowledge'][0]
                    formal_summary += f" including '{top_concept['concept_name']}' (relevance: {top_concept['relevance_score']:.2f})"
                summary_parts.append(formal_summary)

            if result['conversation_insights']['results']:
                conv_count = len(result['conversation_insights']['results'])
                conv_summary = f"Conversation insights: Found {conv_count} relevant discussion fragments"
                summary_parts.append(conv_summary)

            if result['cross_references']:
                cross_summary = f"Cross-references: {len(result['cross_references'])} connections between formal knowledge and conversations"
                summary_parts.append(cross_summary)

            result['unified_summary'] = '. '.join(summary_parts) if summary_parts else "No relevant knowledge found"

            # Calculate overall confidence using semantic scores
            formal_confidence = max([m['relevance_score'] for m in result['formal_knowledge']], default=0.0)
            conv_confidence = 0.6 if result['conversation_insights']['results'] else 0.0
            cross_confidence = max([link['confidence'] for link in result['cross_references']], default=0.0)

            result['confidence_score'] = max(formal_confidence, conv_confidence, cross_confidence)

            logger.info(f"Blended query complete: {len(result['formal_knowledge'])} formal, {len(result['conversation_insights'].get('results', []))} conversational, {len(result['cross_references'])} cross-refs")

            return result

        except Exception as e:
            logger.error(f"Error in blended query: {e}")
            result['unified_summary'] = f"Error processing query: {str(e)}"
            return result

    async def get_concept(self, concept_name: str) -> dict[str, Any] | None:
        """
        Direct concept retrieval from document knowledge.

        Returns structured knowledge about specific concept,
        not conversation fragments.

        Args:
            concept_name: Name of concept to retrieve

        Returns:
            Concept information dict or None if not found
        """
        if not self.initialized:
            await self.initialize()

        self._ensure_component(self.document_store, "document_store")

        try:
            # Search for concept in document store
            document, concept = await self.document_store.get_concept_by_name(concept_name)

            if not document or not concept:
                logger.info(f"Concept '{concept_name}' not found in document knowledge")
                return None

            # Update access time
            document.last_accessed = datetime.now()
            concept.last_updated = datetime.now()

            # Build concept information
            concept_info = {
                "concept_name": concept.name,
                "concept_id": concept.concept_id,
                "description": concept.description,
                "content": concept.content,
                "domain": concept.domain.value,
                "document_context": {
                    "document_id": document.doc_id,
                    "document_title": document.title,
                    "root_concept": document.root_concept
                },
                "hierarchy": {
                    "parent_concept_id": concept.parent_concept_id,
                    "child_concept_ids": concept.child_concept_ids,
                    "related_concept_ids": concept.related_concept_ids
                },
                "structured_data": concept.structured_data,
                "examples": concept.examples,
                "metadata": {
                    "salience_score": concept.salience_score,
                    "confidence_score": concept.confidence_score,
                    "tags": concept.tags,
                    "created": concept.created.isoformat(),
                    "last_updated": concept.last_updated.isoformat()
                },
                "source_type": "formal_knowledge",
                "retrieved_at": datetime.now().isoformat()
            }

            logger.info(f"Retrieved concept '{concept_name}' from document knowledge")
            return concept_info

        except Exception as e:
            logger.error(f"Failed to retrieve concept '{concept_name}': {e}")
            raise CMEError(f"Concept retrieval failed: {e}") from e

    async def browse_knowledge_shelf(self, domain: str) -> dict[str, Any]:
        """
        Browse all concepts in a knowledge domain.

        Returns organized view of all concepts in domain like
        AI_ARCHITECTURE, PROMPT_ENGINEERING, etc.

        Args:
            domain: Knowledge domain to browse

        Returns:
            Knowledge shelf information with documents and concepts
        """
        if not self.initialized:
            await self.initialize()

        self._ensure_component(self.document_store, "document_store")

        try:
            # Convert domain string to enum
            try:
                domain_enum = KnowledgeDomain(domain.lower())
            except ValueError:
                logger.warning(f"Unknown domain '{domain}', listing available domains")
                return {
                    "error": f"Unknown domain '{domain}'",
                    "available_domains": [d.value for d in KnowledgeDomain],
                    "suggestion": "Use one of the available domains listed above"
                }

            logger.info(f"Browsing knowledge shelf for domain {domain_enum.value}")

            # Get knowledge shelf
            shelf = await self.document_store.get_shelf(domain_enum)

            # Get all documents in this domain
            documents = await self.document_store.list_documents_by_domain(domain_enum)

            # Build shelf information
            shelf_info = {
                "domain": domain_enum.value,
                "domain_name": domain_enum.value.replace('_', ' ').title(),
                "total_documents": len(documents),
                "documents": [],
                "featured_concepts": [],
                "browsed_at": datetime.now().isoformat()
            }

            # Add shelf metadata if exists
            if shelf:
                shelf_info.update({
                    "shelf_name": shelf.name,
                    "description": shelf.description,
                    "tags": shelf.tags,
                    "subcategories": shelf.subcategories,
                    "created": shelf.created.isoformat(),
                    "last_accessed": shelf.last_accessed.isoformat()
                })

            # Add document information
            for document in documents:
                doc_info = {
                    "document_id": document.doc_id,
                    "title": document.title,
                    "root_concept": document.root_concept,
                    "total_concepts": document.total_concepts,
                    "created": document.created.isoformat(),
                    "last_accessed": document.last_accessed.isoformat(),
                    "concepts": []
                }

                # Add top-level concepts (direct children of root)
                if document.root_concept_id in document.concepts:
                    root_concept = document.concepts[document.root_concept_id]
                    for child_id in root_concept.child_concept_ids:
                        if child_id in document.concepts:
                            child_concept = document.concepts[child_id]
                            concept_info = {
                                "concept_id": child_concept.concept_id,
                                "name": child_concept.name,
                                "description": child_concept.description[:100] + "..." if len(child_concept.description) > 100 else child_concept.description,
                                "children_count": len(child_concept.child_concept_ids),
                                "salience_score": child_concept.salience_score
                            }
                            doc_info["concepts"].append(concept_info)

                            # Add to featured concepts if high salience
                            if child_concept.salience_score > 0.7:
                                shelf_info["featured_concepts"].append({
                                    "name": child_concept.name,
                                    "document_title": document.title,
                                    "salience_score": child_concept.salience_score
                                })

                shelf_info["documents"].append(doc_info)

            # Sort featured concepts by salience
            shelf_info["featured_concepts"].sort(key=lambda x: x["salience_score"], reverse=True)

            logger.info(f"Retrieved {len(documents)} documents from {domain_enum.value} shelf")
            return shelf_info

        except Exception as e:
            logger.error(f"Failed to browse knowledge shelf for domain '{domain}': {e}")
            raise CMEError(f"Knowledge shelf browsing failed: {e}") from e

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

        self._ensure_component(self.temporal_organizer, "temporal_organizer")
        self._ensure_component(self.context_assembler, "context_assembler")

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

        self._ensure_component(self.context_assembler, "context_assembler")
        self._ensure_component(self.response_generator, "response_generator")

        try:
            logger.info(f"Generating response for: '{prompt[:50]}...'")

            # Step 1: Intelligent query analysis for adaptive memory retrieval
            query_analysis = await self._analyze_query_intent(prompt)

            # Step 2: Query relevant memory context with adaptive parameters
            await self.query_memory(
                prompt,
                query_analysis["context_depth"],
                query_analysis["time_scope"],  # Fixed: time_scope not temporal_scope
                query_analysis["max_results"]
            )

            # Step 3: Assemble context for response generation with optimal strategy
            context = await self.context_assembler.assemble_context(
                query=prompt,
                max_depth=query_analysis["context_depth"],
                temporal_scope=query_analysis["temporal_scope"],
                strategy=query_analysis["retrieval_strategy"]
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

    async def _analyze_query_intent(self, prompt: str) -> dict[str, Any]:
        """
        Analyze query intent to optimize memory retrieval parameters.

        Inspired by procedural memory patterns from Memp paper - different
        query types benefit from different temporal scopes and retrieval strategies.

        Args:
            prompt: User's query text

        Returns:
            Dictionary with optimized retrieval parameters
        """
        prompt_lower = prompt.lower()

        # Query type detection patterns
        debugging_patterns = ["error", "debug", "fix", "broken", "issue", "problem", "troubleshoot", "fail"]
        implementation_patterns = ["how to", "implement", "create", "build", "make", "code", "write"]
        research_patterns = ["what is", "explain", "research", "analyze", "understand", "learn"]
        recent_patterns = ["today", "yesterday", "recent", "latest", "current", "now"]

        # Default parameters (original behavior)
        analysis = {
            "context_depth": 3,
            "temporal_scope": "week",
            "max_results": 5,
            "retrieval_strategy": "hybrid",
            "query_type": "general"
        }

        # Debugging queries - look for recent similar problems and solutions
        if any(pattern in prompt_lower for pattern in debugging_patterns):
            analysis.update({
                "context_depth": 4,  # Deeper context for troubleshooting
                "temporal_scope": "month",  # Look back further for similar issues
                "max_results": 8,  # More examples of solutions
                "retrieval_strategy": "hybrid",
                "query_type": "debugging"
            })

        # Implementation queries - find proven patterns and approaches
        elif any(pattern in prompt_lower for pattern in implementation_patterns):
            analysis.update({
                "context_depth": 5,  # Deep context for implementation details
                "temporal_scope": "all",  # Look for all proven patterns
                "max_results": 7,  # Multiple implementation examples
                "retrieval_strategy": "hybrid",
                "query_type": "implementation"
            })

        # Research queries - broad knowledge retrieval
        elif any(pattern in prompt_lower for pattern in research_patterns):
            analysis.update({
                "context_depth": 3,
                "temporal_scope": "all",  # All available knowledge
                "max_results": 10,  # Comprehensive information
                "retrieval_strategy": "hybrid",
                "query_type": "research"
            })

        # Recent/temporal queries - focus on recency
        elif any(pattern in prompt_lower for pattern in recent_patterns):
            analysis.update({
                "context_depth": 2,
                "temporal_scope": "day",  # Very recent focus
                "max_results": 5,
                "retrieval_strategy": "hybrid",
                "query_type": "recent"
            })

        logger.info(f"Query analysis: {analysis['query_type']} -> scope: {analysis['temporal_scope']}, depth: {analysis['context_depth']}")
        return analysis

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

        self._ensure_component(self.temporal_organizer, "temporal_organizer")

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

        self._ensure_component(self.vector_store, "vector_store")
        self._ensure_component(self.temporal_organizer, "temporal_organizer")
        self._ensure_component(self.temporal_library, "temporal_library")

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

            if include_details and self.narrative_builder and self.response_generator:
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

        self._ensure_component(self.temporal_library, "temporal_library")

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
        if not self.initialized:
            await self.initialize()
        self._ensure_component(self.rtm_store, "rtm_store")
        return await self.rtm_store.list_trees(limit=20)

    async def get_temporal_organization(self) -> dict[str, Any]:
        """Get temporal organization structure."""
        if not self.initialized:
            await self.initialize()
        self._ensure_component(self.temporal_organizer, "temporal_organizer")
        return await self.temporal_organizer.get_statistics()

    async def get_active_context(self) -> dict[str, Any]:
        """Get current active workspace context."""
        if not self.initialized:
            await self.initialize()
        self._ensure_component(self.context_assembler, "context_assembler")

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

            # Cleanup dual-track components
            if self.document_store:
                # Document store doesn't have async cleanup, but we can clear caches
                self.document_store.documents.clear()
                self.document_store.shelves.clear()

            if self.cross_reference_store:
                # Cross-reference store cleanup
                self.cross_reference_store.links.clear()

            self.active_sessions.clear()
            self.initialized = False
            logger.info("Cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
