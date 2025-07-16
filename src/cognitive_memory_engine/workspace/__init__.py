"""
Active Workspace Module

Components for managing the dynamic memory workspace that bridges
long-term storage and short-term response generation.
"""

from .context_assembler import ContextAssembler
from .vector_manager import VectorManager

__all__ = [
    "VectorManager",
    "ContextAssembler"
]
