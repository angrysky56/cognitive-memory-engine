"""
Enhanced Knowledge Tools for Cognitive Memory Engine

This package provides enhanced knowledge ingestion capabilities for the
cognitive memory engine, including:

- Direct URL content fetching with intelligent merging
- Search-based knowledge aggregation 
- Existing concept enhancement with multiple sources
- Automatic source type detection and processing
"""

from .enhanced_server_tools import (
    EnhancedKnowledgeServerTools,
    initialize_enhanced_knowledge_tools,
    ENHANCED_KNOWLEDGE_TOOL_DEFINITIONS
)

__all__ = [
    'EnhancedKnowledgeServerTools',
    'initialize_enhanced_knowledge_tools', 
    'ENHANCED_KNOWLEDGE_TOOL_DEFINITIONS'
]

__version__ = "1.0.0"
