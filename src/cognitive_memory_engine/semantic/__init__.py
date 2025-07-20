"""
Semantic Enhancement Module - Phase 3A Foundation

Provides abstract interfaces for compositional semantic understanding,
transforming the cognitive memory engine from statistical pattern matching
to formal logic-based reasoning using Montague Grammar principles.

Key Components:
- AbstractSemanticProcessor: Base interface for semantic analysis
- AbstractGraphBuilder: Knowledge graph construction interface
- AbstractQueryEngine: Logical graph traversal interface

This module extends the existing dual-track architecture with semantic intelligence
while maintaining backward compatibility with Phase 2 systems.
"""

from .abstract_processors import (
    AbstractGraphBuilder,
    AbstractQueryEngine,
    AbstractSemanticProcessor,
    GraphConstructionError,
    QueryExecutionError,
    SemanticProcessingError,
)

__all__ = [
    "AbstractSemanticProcessor",
    "AbstractGraphBuilder",
    "AbstractQueryEngine",
    "SemanticProcessingError",
    "GraphConstructionError",
    "QueryExecutionError",
]
from .logical_query_processor import LogicalQueryProcessor, LQEPRResult

__all__ = ['LogicalQueryProcessor', 'LQEPRResult']
