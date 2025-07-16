"""
Core data structures for the Cognitive Memory Engine

Defines the fundamental types used across all modules, inspired by:
- Random Tree Model (RTM) for narrative hierarchies
- Temporal organization for time-based memory
- Neural gain mechanism for salience weighting
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TemporalScale(Enum):
    """Temporal scales for hierarchical memory organization"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class NodeType(Enum):
    """Types of nodes in RTM narrative trees"""
    LEAF = "leaf"           # Original conversation segments
    SUMMARY = "summary"     # Intermediate compression nodes
    ROOT = "root"          # Top-level narrative gist
    THEME = "theme"        # Persistent themes across time


class ShelfCategory(Enum):
    """Categories for temporal book organization"""
    ACTIVE = "active"       # Currently developing (< 1 day)
    RECENT = "recent"       # Recently completed (1-7 days)
    REFERENCE = "reference" # Persistent themes (any age)
    ARCHIVED = "archived"   # Older memories (> 30 days)


@dataclass
class ConversationTurn:
    """A single turn in a conversation"""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RTMNode:
    """A node in a Random Tree Model narrative hierarchy"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tree_id: str = ""
    content: str = ""
    summary: str = ""
    node_type: NodeType = NodeType.LEAF

    # Hierarchical structure
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0

    # Neural gain properties (from IEM research)
    base_embedding: list[float] | None = None
    salience_score: float = 1.0  # Neural gain multiplier
    weighted_embedding: list[float] | None = None

    # Temporal properties
    timestamp: datetime = field(default_factory=datetime.now)
    temporal_scale: TemporalScale = TemporalScale.MINUTE

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_salient_embedding(self) -> list[float] | None:
        """Get embedding weighted by neural gain (salience)"""
        if self.base_embedding is None:
            return None
        return [x * self.salience_score for x in self.base_embedding]


@dataclass
class RTMTree:
    """A complete Random Tree Model narrative structure"""
    tree_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""

    # Tree structure
    root_node_id: str = ""
    nodes: dict[str, RTMNode] = field(default_factory=dict)

    # RTM parameters (from research)
    max_branching_factor: int = 4   # K parameter
    max_recall_depth: int = 6       # D parameter

    # Compression statistics
    total_clauses: int = 0
    compression_ratio: float = 1.0
    node_count: int = 0

    # Temporal properties
    created: datetime = field(default_factory=datetime.now)
    temporal_scale: TemporalScale = TemporalScale.DAY

    # Session tracking
    session_id: str = ""

    def get_nodes_at_depth(self, depth: int) -> list[RTMNode]:
        """Get all nodes at a specific depth in the tree"""
        return [node for node in self.nodes.values() if node.depth == depth]

    def traverse_to_depth(self, max_depth: int) -> list[RTMNode]:
        """Traverse tree up to specified depth, returning visited nodes"""
        if not self.root_node_id or self.root_node_id not in self.nodes:
            return []

        visited = []
        queue = [(self.root_node_id, 0)]

        while queue:
            node_id, depth = queue.pop(0)
            if depth > max_depth:
                continue

            node = self.nodes[node_id]
            visited.append(node)

            # Add children to queue
            for child_id in node.children_ids:
                if child_id in self.nodes:
                    queue.append((child_id, depth + 1))

        return visited


@dataclass
class TemporalBook:
    """A temporal container for related RTM trees"""
    book_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""

    # Temporal organization
    temporal_scale: TemporalScale = TemporalScale.DAY
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    # Content
    rtm_tree_ids: list[str] = field(default_factory=list)
    persistent_themes: list[str] = field(default_factory=list)

    # Compression properties
    compression_ratio: float = 1.0
    narrative_depth: int = 1

    # Shelf organization
    shelf_category: ShelfCategory = ShelfCategory.ACTIVE
    tags: list[str] = field(default_factory=list)

    # Metadata
    session_ids: list[str] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class LibraryShelf:
    """A shelf organizing related temporal books"""
    shelf_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Organization
    category: ShelfCategory = ShelfCategory.ACTIVE
    book_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Temporal scope
    primary_scale: TemporalScale = TemporalScale.DAY
    time_span_start: datetime = field(default_factory=datetime.now)
    time_span_end: datetime | None = None

    # Metadata
    created: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalContext:
    """Context assembled for response generation"""
    query: str = ""
    retrieved_nodes: list[RTMNode] = field(default_factory=list)
    temporal_books_accessed: list[str] = field(default_factory=list)

    # Salience properties
    max_salience_score: float = 0.0
    avg_salience_score: float = 0.0
    total_context_length: int = 0

    # Social context
    trust_score: float = 1.0
    formality_level: float = 0.5
    emotional_context: dict[str, float] = field(default_factory=dict)

    # Temporal context
    conversation_history_depth: int = 1
    temporal_scope: TemporalScale | None = None

    # Reasoning
    retrieval_strategy: str = "hybrid"
    reasoning_chain: list[str] = field(default_factory=list)


@dataclass
class GeneratedResponse:
    """Response generated by the production module"""
    content: str = ""
    confidence: float = 1.0

    # Context used
    context_nodes_used: int = 0
    context_salience_threshold: float = 0.0

    # Social modulation
    trust_adjusted: bool = False
    formality_adjusted: bool = False

    # Reasoning
    reasoning_chain: list[str] = field(default_factory=list)
    prediction_errors: list[str] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    generation_time_ms: int = 0


@dataclass
class NeuralGainConfig:
    """Configuration for neural gain mechanism"""
    base_salience: float = 1.0
    temporal_decay_factor: float = 0.1
    relevance_threshold: float = 0.5
    max_gain_multiplier: float = 3.0

    # Compression targets per temporal scale
    compression_targets: dict[TemporalScale, float] = field(default_factory=lambda: {
        TemporalScale.MINUTE: 1.0,
        TemporalScale.HOUR: 1.2,
        TemporalScale.DAY: 2.0,
        TemporalScale.WEEK: 4.0,
        TemporalScale.MONTH: 8.0,
        TemporalScale.YEAR: 16.0
    })


@dataclass
class RTMConfig:
    """Configuration for Random Tree Model parameters"""
    max_branching_factor: int = 4      # K parameter
    max_recall_depth: int = 6          # D parameter
    min_segment_size: int = 1          # Minimum clauses per leaf
    max_summary_length: int = 150      # Max characters in summary

    # Persistence rules
    persistence_threshold: float = 0.7
    theme_extraction_threshold: float = 0.8

    # Social governance
    trust_decay_rate: float = 0.05
    formality_adaptation_rate: float = 0.1


@dataclass
class MemoryQuery:
    """A query to the memory system"""
    query: str
    max_context_depth: int = 3
    temporal_scope: str | None = None  # 'hour', 'day', 'week', etc.
    include_social_context: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration"""
    data_directory: str
    llm_model: str
    embedding_model: str
    rtm_config: RTMConfig
    neural_gain_config: NeuralGainConfig
    max_context_length: int
    vector_similarity_threshold: float
    auto_archive_days: int
    vector_manager: str = "chroma"
    svg_config: dict[str, Any] = field(default_factory=dict)
