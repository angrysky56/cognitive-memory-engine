"""
Abstract Semantic Processor Interfaces - Phase 3A Foundation

Provides the engineering backbone for semantic enhancement, implementing
abstract base classes that ensure consistent interfaces across all
semantic processing components.

Following AETHELRED principles:
- Prioritize Clarity: Clear method signatures and documentation
- Embrace Simplicity: Minimal, focused abstract interfaces
- Think Structurally: Consistent patterns across semantic processors
- Be Collaborative: Extensible design for future enhancement
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# Custom exceptions for semantic processing
class SemanticProcessingError(Exception):
    """Base exception for semantic processing errors"""
    pass

class GraphConstructionError(SemanticProcessingError):
    """Exception raised when knowledge graph construction fails"""
    pass

class QueryExecutionError(SemanticProcessingError):
    """Exception raised when logical query execution fails"""
    pass

@dataclass
class SemanticResult:
    """Standard result format for semantic processing operations"""
    success: bool
    data: Any
    metadata: dict[str, Any]
    processing_time: float
    timestamp: datetime
    error_message: str | None = None

@dataclass
class LogicalForm:
    """Represents a logical form generated from natural language"""
    expression: str  # Lambda calculus or first-order logic
    variables: list[str]
    quantifiers: list[str]
    predicates: list[str]
    confidence: float
    source_text: str

@dataclass
class SemanticNode:
    """Node in the semantic knowledge graph"""
    node_id: str
    entity_name: str
    semantic_type: str  # From Montague classification
    domain: str
    logical_form: LogicalForm | None
    properties: dict[str, Any]
    relationships: list[str]  # IDs of related nodes

@dataclass
class SemanticEdge:
    """Edge in the semantic knowledge graph"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    logical_formula: str  # Formal relationship definition
    confidence: float
    created_at: datetime


class AbstractSemanticProcessor(ABC):
    """
    Abstract base class for all semantic analysis processors.

    Provides consistent interface for semantic understanding components
    implementing Montague Grammar and compositional semantics.

    Following AETHELRED principles:
    - Clear abstract methods for semantic analysis
    - Consistent error handling patterns
    - Performance monitoring hooks
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.processing_stats = {
            "total_operations": 0,
            "total_processing_time": 0.0,
            "error_count": 0
        }

    @abstractmethod
    async def parse_semantics(self, text: str) -> SemanticResult:
        """
        Parse natural language text into semantic representation.

        Args:
            text: Natural language input to analyze

        Returns:
            SemanticResult containing parsed semantic information

        Raises:
            SemanticProcessingError: If parsing fails
        """
        pass

    @abstractmethod
    async def extract_relations(self, semantic_data: Any) -> list[tuple[str, str, str]]:
        """
        Extract entity relationships from semantic data.

        Args:
            semantic_data: Parsed semantic representation

        Returns:
            list of (subject, predicate, object) relationship triples

        Raises:
            SemanticProcessingError: If relation extraction fails
        """
        pass

    @abstractmethod
    async def validate_logic(self, logical_form: LogicalForm) -> bool:
        """
        Validate logical consistency of extracted semantic forms.

        Args:
            logical_form: Logical representation to validate

        Returns:
            True if logically consistent, False otherwise

        Raises:
            SemanticProcessingError: If validation process fails
        """
        pass

    def update_stats(self, operation_time: float, success: bool):
        """Update processing statistics"""
        self.processing_stats["total_operations"] += 1
        self.processing_stats["total_processing_time"] += operation_time
        if not success:
            self.processing_stats["error_count"] += 1

    def get_performance_metrics(self) -> dict[str, float]:
        """Get current performance metrics"""
        total_ops = self.processing_stats["total_operations"]
        if total_ops == 0:
            return {"avg_processing_time": 0.0, "success_rate": 1.0}

        return {
            "avg_processing_time": self.processing_stats["total_processing_time"] / total_ops,
            "success_rate": 1.0 - (self.processing_stats["error_count"] / total_ops)
        }


class AbstractGraphBuilder(ABC):
    """
    Abstract base class for knowledge graph construction.

    Provides consistent interface for building semantic knowledge graphs
    with DuckDB backend integration and batch processing capabilities.

    Following AETHELRED principles:
    - Structured graph construction patterns
    - Performance-optimized batch operations
    - Clear validation and error handling
    """

    def __init__(self, db_connection: Any, config: dict[str, Any]):
        self.db_connection = db_connection
        self.config = config
        self.batch_size = config.get("batch_size", 1000)

    @abstractmethod
    async def build_nodes(self, semantic_data: list[Any]) -> list[SemanticNode]:
        """
        Build semantic nodes from processed semantic data.

        Args:
            semantic_data: list of semantic representations

        Returns:
            list of SemanticNode objects

        Raises:
            GraphConstructionError: If node building fails
        """
        pass

    @abstractmethod
    async def create_edges(self, nodes: list[SemanticNode]) -> list[SemanticEdge]:
        """
        Create semantic edges between nodes based on relationships.

        Args:
            nodes: list of semantic nodes to connect

        Returns:
            list of SemanticEdge objects

        Raises:
            GraphConstructionError: If edge creation fails
        """
        pass

    @abstractmethod
    async def validate_graph(self, nodes: list[SemanticNode],
                           edges: list[SemanticEdge]) -> bool:
        """
        Validate logical consistency of constructed graph.

        Args:
            nodes: list of semantic nodes
            edges: list of semantic edges

        Returns:
            True if graph is logically consistent

        Raises:
            GraphConstructionError: If validation fails
        """
        pass

    async def batch_process(self, data: list[Any],
                          process_func: Callable) -> list[Any]:
        """
        Process data in batches for performance optimization.

        Args:
            data: Data to process
            process_func: Function to apply to each batch

        Returns:
            Combined results from all batches
        """
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = await process_func(batch)
            results.extend(batch_result)
        return results


class AbstractQueryEngine(ABC):
    """
    Abstract base class for logical graph traversal and querying.

    Provides consistent interface for executing compositional queries
    against semantic knowledge graphs using formal logic.

    Following AETHELRED principles:
    - Clear query composition interfaces
    - Caching and optimization hooks
    - Transparent performance monitoring
    """

    def __init__(self, graph_store: Any, config: dict[str, Any]):
        self.graph_store = graph_store
        self.config = config
        self.query_cache = {}
        self.cache_enabled = config.get("enable_caching", True)

    @abstractmethod
    async def traverse_graph(self, start_nodes: list[str],
                           query_pattern: str) -> list[str]:
        """
        Traverse semantic graph following logical query pattern.

        Args:
            start_nodes: Node IDs to begin traversal
            query_pattern: Logical pattern to follow

        Returns:
            list of matching node IDs

        Raises:
            QueryExecutionError: If traversal fails
        """
        pass

    @abstractmethod
    async def compose_logic(self, sub_queries: list[str]) -> str:
        """
        Compose multiple logical sub-queries into unified query.

        Args:
            sub_queries: list of logical sub-query expressions

        Returns:
            Composed logical query expression

        Raises:
            QueryExecutionError: If composition fails
        """
        pass

    @abstractmethod
    async def rank_results(self, results: list[str],
                         query_context: dict[str, Any]) -> list[tuple[str, float]]:
        """
        Rank query results by logical relevance and confidence.

        Args:
            results: list of result node IDs
            query_context: Context for ranking decisions

        Returns:
            list of (node_id, relevance_score) tuples

        Raises:
            QueryExecutionError: If ranking fails
        """
        pass

    async def cached_query(self, query_key: str, query_func: Callable) -> Any:
        """
        Execute query with caching support.

        Args:
            query_key: Unique key for caching
            query_func: Query function to execute

        Returns:
            Query results (cached or fresh)
        """
        if self.cache_enabled and query_key in self.query_cache:
            return self.query_cache[query_key]

        result = await query_func()

        if self.cache_enabled:
            self.query_cache[query_key] = result

        return result

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.query_cache),
            "cache_enabled": self.cache_enabled
        }
