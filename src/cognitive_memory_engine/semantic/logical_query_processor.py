"""
LQEPR (Logical Query Enhanced Pattern Retrieval) Unified Query Interface

Unifies Prolog formal logic + DuckDB graph queries + ChromaDB
vector retrieval for comprehensive semantic search and reasoning.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..storage.semantic_graph_store import SemanticGraphStore
from ..storage.vector_store import VectorStore
from ..types import LogicalResult, QueryMode
from .abstract_processors import AbstractQueryEngine, QueryExecutionError
from .prolog_processor import PrologSemanticProcessor

logger = logging.getLogger(__name__)


@dataclass
class LQEPRResult:
    """Unified result from LQEPR query interface"""

    query_text: str
    query_mode: QueryMode
    logical_results: list[LogicalResult]
    graph_results: list[tuple[str, float]]
    vector_results: list[tuple[str, float]]
    unified_score: float
    confidence: float
    execution_time: float
    query_metadata: dict[str, Any]

    def to_rich_table(self) -> Any | None:
        """Convert results to rich table for CLI display"""
        if not Table:
            return None

        table = Table(title=f"LQEPR Query Results: {self.query_text[:50]}...")
        table.add_column("Source", style="cyan")
        table.add_column("Result", style="white")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Type", style="yellow")

        # Add logical results
        for result in self.logical_results[:3]:
            table.add_row(
                "Prolog Logic",
                str(result.logical_form)[:60],
                f"{result.confidence:.3f}",
                "Formal Logic"
            )

        # Add graph results
        for node_id, score in self.graph_results[:3]:
            table.add_row(
                "DuckDB Graph",
                node_id[:60],
                f"{score:.3f}",
                "Graph Query"
            )

        # Add vector results
        for content, score in self.vector_results[:3]:
            table.add_row(
                "ChromaDB Vector",
                str(content)[:60],
                f"{score:.3f}",
                "Vector Similarity"
            )

        return table


class LogicalQueryProcessor(AbstractQueryEngine):
    """
    LQEPR Unified Query Interface - Phase 3B-2 Implementation

    Integrates three query modes for comprehensive semantic retrieval:
    1. Formal Logic (Prolog) - compositional reasoning and rule-based inference
    2. Graph Traversal (DuckDB) - relationship queries and structural analysis
    3. Vector Similarity (ChromaDB) - neural gain retrieval and semantic matching

    Optimized for CLI usage with rich formatting and interactive features.
    """

    def __init__(self,
                 prolog_processor: PrologSemanticProcessor | None = None,
                 semantic_graph_store: SemanticGraphStore | None = None,
                 vector_store: VectorStore | None = None,
                 config: dict[str, Any] | None = None):
        """
        Initialize LQEPR unified query processor.

        Args:
            prolog_processor: Formal logic reasoning engine
            semantic_graph_store: DuckDB semantic graph store
            vector_store: ChromaDB vector store with neural gain
            config: Configuration dict for caching, timeouts, etc.
        """
        super().__init__(graph_store=semantic_graph_store, config=config or {})

        self.prolog_processor = prolog_processor
        self.semantic_graph_store = semantic_graph_store
        self.vector_store = vector_store
        self.console = Console() if Console else None

# Query mode availability
        self.modes_available = {
            QueryMode.LOGICAL: prolog_processor is not None,
            QueryMode.GRAPH: semantic_graph_store is not None,
            QueryMode.VECTOR: vector_store is not None
        }

        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'avg_execution_time': 0.0,
            'mode_usage': dict.fromkeys(QueryMode, 0),
            'cache_hits': 0
        }

        logger.info(f"LQEPR initialized. Available modes: {[mode.value for mode, avail in self.modes_available.items() if avail]}")

    async def unified_query(self,
                          query_text: str,
                          modes: list[QueryMode] | None = None,
                          max_results: int = 10,
                          timeout: float = 30.0,
                          show_progress: bool = True) -> LQEPRResult:
        """
        Execute unified query across multiple backends with optional progress display.

        Args:
            query_text: Natural language query
            modes: Query modes to use (default: all available)
            max_results: Maximum results per mode
            timeout: Query timeout in seconds
            show_progress: Show rich progress bar for CLI usage

        Returns:
            LQEPRResult with unified results from all modes

        Raises:
            QueryExecutionError: If all query modes fail
        """
        start_time = datetime.now()

        if modes is None:
            modes = [mode for mode, available in self.modes_available.items() if available]

        if not modes:
            raise QueryExecutionError("No query modes available")

        # Initialize result containers
        logical_results = []
        graph_results = []
        vector_results = []

        # Progress tracking for CLI
        if show_progress and self.console is not None:
            from rich.progress import Progress, SpinnerColumn, TextColumn

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )

            with progress:
                task = progress.add_task("Executing LQEPR unified query...", total=len(modes))

                # Execute each query mode
                for mode in modes:
                    progress.update(task, description=f"Querying {mode.value}...")

                    try:
                        if mode == QueryMode.LOGICAL and self.prolog_processor:
                            logical_results = await self._query_logical(query_text, max_results)
                        elif mode == QueryMode.GRAPH and self.semantic_graph_store:
                            graph_results = await self._query_graph(query_text, max_results)
                        elif mode == QueryMode.VECTOR and self.vector_store:
                            vector_results = await self._query_vector(query_text, max_results)
                    except Exception as e:
                        logger.warning(f"Query mode {mode.value} failed: {e}")

                    progress.advance(task)
        else:
            # Execute without progress display
            try:
                if QueryMode.LOGICAL in modes and self.prolog_processor:
                    logical_results = await self._query_logical(query_text, max_results)
                if QueryMode.GRAPH in modes and self.semantic_graph_store:
                    graph_results = await self._query_graph(query_text, max_results)
                if QueryMode.VECTOR in modes and self.vector_store:
                    vector_results = await self._query_vector(query_text, max_results)
            except Exception as e:
                logger.error(f"Unified query execution failed: {e}")

        # Calculate unified scores and ranking
        unified_score = self._calculate_unified_score(logical_results, graph_results, vector_results)
        confidence = self._calculate_confidence(logical_results, graph_results, vector_results)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Update statistics
        self.query_stats['total_queries'] += 1
        self.query_stats['avg_execution_time'] = (
            (self.query_stats['avg_execution_time'] * (self.query_stats['total_queries'] - 1) + execution_time)
            / self.query_stats['total_queries']
        )

        for mode in modes:
            self.query_stats['mode_usage'][mode] += 1

        return LQEPRResult(
            query_text=query_text,
            query_mode=QueryMode.UNIFIED,
            logical_results=logical_results,
            graph_results=graph_results,
            vector_results=vector_results,
            unified_score=unified_score,
            confidence=confidence,
            execution_time=execution_time,
            query_metadata={'modes_used': [m.value for m in modes]}
        )

    async def _query_logical(self, query_text: str, max_results: int) -> list[LogicalResult]:
        """
        Execute Prolog formal logic query

        Args:
            query_text: Query text
            max_results: Maximum results to return

        Returns:
            List of LogicalResult objects

        Raises:
            QueryExecutionError: If Prolog is not available
        """
        if not self.prolog_processor:
            raise QueryExecutionError(
                "Prolog processor not available. Install SWI-Prolog: sudo apt-get install swi-prolog"
            )

        if not hasattr(self.prolog_processor, 'query_compositional'):
            raise QueryExecutionError(
                "Prolog processor does not support compositional queries"
            )

        return await self.prolog_processor.query_compositional(query_text, max_results)

    async def _query_graph(self, query_text: str, max_results: int) -> list[tuple[str, float]]:
        """Execute DuckDB graph traversal query"""
        try:
            # Use semantic graph store to find related concepts
            if not self.semantic_graph_store:
                raise QueryExecutionError("Semantic graph store not available")
            concepts = await self.semantic_graph_store.search_concepts(query_text, limit=max_results)
            return [(concept['concept_id'], concept.get('relevance_score', 0.5)) for concept in concepts]
        except Exception as e:
            logger.warning(f"Graph query failed: {e}")
            return []

    async def _query_vector(self, query_text: str, max_results: int) -> list[tuple[str, float]]:
        """Execute ChromaDB vector similarity query"""
        try:
            if not self.vector_store:
                raise QueryExecutionError("Vector store not available")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query_text).tolist()
            results = await self.vector_store.search_vectors(query_embedding=query_embedding, n_results=max_results)
            return [(result[0], float(result[1])) for result in results]
        except Exception as e:
            logger.warning(f"Vector query failed: {e}")
            return []

    def _calculate_unified_score(self, logical: list, graph: list, vector: list) -> float:
        """Calculate unified relevance score from all query modes"""
        scores = []

        if logical:
            scores.append(sum(r.confidence for r in logical) / len(logical))
        if graph:
            scores.append(sum(score for _, score in graph) / len(graph))
        if vector:
            scores.append(sum(score for _, score in vector) / len(vector))

        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_confidence(self, logical: list, graph: list, vector: list) -> float:
        """Calculate overall confidence based on result agreement"""
        # Simple confidence calculation - more sophisticated logic possible
        mode_count = sum(1 for results in [logical, graph, vector] if results)
        return min(0.9, mode_count * 0.3) if mode_count > 0 else 0.1

    async def traverse_graph(self, start_nodes: list[str], query_pattern: str) -> list[str]:
        """Traverse semantic graph following logical query pattern"""
        if not self.semantic_graph_store:
            raise QueryExecutionError("Semantic graph store not available")

        try:
            # Use DuckDB to traverse relationships
            results = await self.semantic_graph_store.traverse_relationships(
                start_nodes, query_pattern
            )
            return [result['target_concept_id'] for result in results]
        except Exception as e:
            raise QueryExecutionError(f"Graph traversal failed: {e}") from e

    async def compose_logic(self, sub_queries: list[str]) -> str:
        """Compose multiple logical sub-queries into unified query"""
        if not self.prolog_processor:
            raise QueryExecutionError("Prolog processor not available")

        try:
            return await self.prolog_processor.compose_queries(sub_queries)
        except Exception as e:
            raise QueryExecutionError(f"Logic composition failed: {e}") from e

    async def rank_results(self, results: list[str], query_context: dict[str, Any]) -> list[tuple[str, float]]:
        """Rank query results by logical relevance and confidence"""
        # Implement sophisticated ranking using all available modes
        ranked_results = []

        for result in results:
            # Calculate composite score from multiple modes
            logical_score = 0.0
            graph_score = 0.0
            vector_score = 0.0

            # Query each mode for this specific result
            if self.prolog_processor:
                logical_score = await self._get_logical_relevance(result, query_context)
            if self.semantic_graph_store:
                graph_score = await self._get_graph_relevance(result, query_context)
            if self.vector_store:
                vector_score = await self._get_vector_relevance(result, query_context)

            # Weighted composite score
            composite_score = (logical_score * 0.4 + graph_score * 0.3 + vector_score * 0.3)
            ranked_results.append((result, composite_score))

        # Sort by composite score descending
        return sorted(ranked_results, key=lambda x: x[1], reverse=True)

    async def _get_logical_relevance(self, result: str, context: dict[str, Any]) -> float:
        """Get logical relevance score from Prolog"""
        if not self.prolog_processor:
            raise QueryExecutionError("Prolog processor not available for logical relevance scoring")

        try:
            # Query Prolog knowledge base for logical relevance
            query = f"relevance('{result}', '{context.get('query', '')}', Score)"
            prolog_results = self.prolog_processor.query_knowledge_base(query)

            if prolog_results and len(prolog_results) > 0:
                # Extract score from Prolog result
                score = prolog_results[0].get('Score', 0.5)
                return float(score) if isinstance(score, int | float) else 0.5
            else:
                # No specific relevance found, return default
                return 0.5

        except Exception as e:
            logger.warning(f"Prolog relevance query failed: {e}")
            raise QueryExecutionError(f"Prolog relevance scoring failed: {e}") from e

    async def _get_graph_relevance(self, result: str, context: dict[str, Any]) -> float:
        """Get graph relevance score from DuckDB"""
        if not self.semantic_graph_store:
            raise QueryExecutionError("Semantic graph store not available for graph relevance scoring")

        try:
            # Search for related concepts in the semantic graph
            query_text = context.get('query', '')
            concepts = await self.semantic_graph_store.search_concepts(query_text, limit=10)

            # Calculate relevance based on graph connections
            relevance_score = 0.0
            for concept in concepts:
                concept_id = concept.get('concept_id', '')
                # Check if result matches or relates to this concept
                if result.lower() in concept_id.lower() or concept_id.lower() in result.lower():
                    relevance_score = max(relevance_score, concept.get('relevance_score', 0.5))

            return min(1.0, relevance_score)

        except Exception as e:
            logger.warning(f"Graph relevance query failed: {e}")
            raise QueryExecutionError(f"Graph relevance scoring failed: {e}") from e

    async def _get_vector_relevance(self, result: str, context: dict[str, Any]) -> float:
        """Get vector relevance score from ChromaDB"""
        if not self.vector_store:
            raise QueryExecutionError("Vector store not available for vector relevance scoring")

        try:
            # Generate embedding for the result text
            from sentence_transformers import SentenceTransformer

            # Use a lightweight model for relevance scoring
            model = SentenceTransformer('all-MiniLM-L6-v2')
            result_embedding = model.encode(result).tolist()

            # Search for similar vectors
            search_results = await self.vector_store.search_vectors(
                query_embedding=result_embedding,
                n_results=5,
                include=['distances', 'metadatas']
            )

            # Calculate relevance based on vector similarity
            if search_results and 'distances' in search_results:
                distances = search_results['distances'][0] if search_results['distances'] else []
                if distances:
                    # Convert distance to similarity (lower distance = higher similarity)
                    max_distance = max(distances) if distances else 1.0
                    similarity = 1.0 - (min(distances) / max_distance)
                    return min(1.0, max(0.0, similarity))

            return 0.3  # Default relevance score

        except Exception as e:
            logger.warning(f"Vector relevance query failed: {e}")
            raise QueryExecutionError(f"Vector relevance scoring failed: {e}") from e

    def display_results(self, result: LQEPRResult, detailed: bool = False) -> None:
        """Display LQEPR results using rich formatting for CLI"""
        if not self.console:
            # Fallback to simple print
            print(f"Query: {result.query_text}")
            print(f"Score: {result.unified_score:.3f}, Confidence: {result.confidence:.3f}")
            print(f"Execution time: {result.execution_time:.3f}s")
            return

        # Rich console display
        if Panel:
            self.console.print(Panel(
                f"[bold blue]LQEPR Query Results[/bold blue]\n"
                f"Query: [white]{result.query_text}[/white]\n"
                f"Score: [green]{result.unified_score:.3f}[/green] | "
                f"Confidence: [yellow]{result.confidence:.3f}[/yellow] | "
                f"Time: [cyan]{result.execution_time:.3f}s[/cyan]",
                title="Unified Query Results"
            ))
        else:
            self.console.print(
                f"LQEPR Query Results\n"
                f"Query: {result.query_text}\n"
                f"Score: {result.unified_score:.3f} | "
                f"Confidence: {result.confidence:.3f} | "
                f"Time: {result.execution_time:.3f}s"
            )

        if detailed:
            table = result.to_rich_table()
            if table is not None:
                self.console.print(table)

    def get_stats(self) -> dict[str, Any]:
        """Get query processor statistics for monitoring"""
        return {
            **self.query_stats,
            'available_modes': [mode.value for mode, avail in self.modes_available.items() if avail],
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self.query_cache)
        }
