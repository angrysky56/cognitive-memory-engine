"""
Comprehension Module

Long-timescale integration components for the Cognitive Memory Engine.
Implements the "listening ear" of the asymmetric architecture.
"""

from .narrative_tree_builder import NarrativeTreeBuilder
from .temporal_organizer import TemporalOrganizer

__all__ = [
    "NarrativeTreeBuilder",
    "TemporalOrganizer"
]
