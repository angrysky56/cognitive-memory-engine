"""
Cognitive Memory Engine - Main Package

Implements a neuroscience-inspired AI memory system combining:
- Random Tree Model (RTM) for narrative hierarchies
- Temporal Books & Shelves for time-based organization
- Asymmetric neural architecture for comprehension vs production
- Neural gain mechanism for salience-weighted retrieval
"""

# Version information
from .__about__ import __version__
from .core.engine import CognitiveMemoryEngine
from .core.exceptions import (
    CMEConfigurationError,
    CMEError,
    CMEInitializationError,
    CMENarrativeError,
    CMEResponseError,
    CMEStorageError,
    CMETemporalError,
    CMEVectorError,
)
from .types import (
    ConversationTurn,
    GeneratedResponse,
    LibraryShelf,
    MemoryQuery,
    NeuralGainConfig,
    NodeType,
    RetrievalContext,
    RTMConfig,
    RTMNode,
    RTMTree,
    ShelfCategory,
    SystemConfig,
    TemporalBook,
    TemporalScale,
)

# Export all key classes and types
__all__ = [
    # Main engine
    "CognitiveMemoryEngine",

    # Core types
    "ConversationTurn", "MemoryQuery", "RTMTree", "RTMNode",
    "TemporalBook", "LibraryShelf", "RetrievalContext",
    "GeneratedResponse", "SystemConfig", "RTMConfig", "NeuralGainConfig",

    # Enums
    "TemporalScale", "NodeType", "ShelfCategory",

    # Exceptions
    "CMEError", "CMEInitializationError", "CMEStorageError",
    "CMEVectorError", "CMETemporalError", "CMENarrativeError",
    "CMEResponseError", "CMEConfigurationError",

    # Version
    "__version__",
]
