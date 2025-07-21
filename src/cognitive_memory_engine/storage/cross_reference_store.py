"""
Cross-Reference Storage Module

Manages persistent storage of cross-references between conversation
and document knowledge tracks.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import ConceptLink, LinkRelationship

logger = logging.getLogger(__name__)


class CrossReferenceStore:
    """
    Persistent storage for concept links between conversations and documents.

    Stores bidirectional references to enable:
    - Finding all conversations that discuss a concept
    - Finding all concepts mentioned in a conversation
    - Tracking relationship types and confidence scores
    """

    def __init__(self, storage_path: str):
        """
        Initialize cross-reference store.

        Args:
            storage_path: Base directory for cross-reference storage
        """
        self.storage_path = Path(storage_path)
        self.links_dir = self.storage_path / "cross_references"
        self.index_file = self.storage_path / "cross_reference_index.json"

        # In-memory indices for fast lookup
        self.conversation_to_concepts: dict[str, list[str]] = {}
        self.concept_to_conversations: dict[str, list[str]] = {}
        self.links: dict[str, ConceptLink] = {}

        self._ensure_directories()
        self._load_index()

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.links_dir.mkdir(exist_ok=True)

    def _load_index(self) -> None:
        """Load existing index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    data = json.load(f)
                    self.conversation_to_concepts = data.get("conversation_to_concepts", {})
                    self.concept_to_conversations = data.get("concept_to_conversations", {})

                # Load individual link files
                for link_file in self.links_dir.glob("*.json"):
                    try:
                        with open(link_file) as f:
                            link_data = json.load(f)
                            link = self._deserialize_link(link_data)
                            self.links[link.link_id] = link
                    except Exception as e:
                        logger.warning(f"Failed to load link {link_file}: {e}")

            except Exception as e:
                logger.warning(f"Failed to load cross-reference index: {e}")

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            index_data = {
                "conversation_to_concepts": self.conversation_to_concepts,
                "concept_to_conversations": self.concept_to_conversations,
                "last_updated": datetime.now().isoformat()
            }

            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save cross-reference index: {e}")

    def _serialize_link(self, link: ConceptLink) -> dict[str, Any]:
        """Convert ConceptLink to JSON-serializable format."""
        return {
            "link_id": link.link_id,
            "conversation_node_id": link.conversation_node_id,
            "conversation_tree_id": link.conversation_tree_id,
            "document_concept_id": link.document_concept_id,
            "document_id": link.document_id,
            "relationship_type": link.relationship_type.value,
            "confidence_score": link.confidence_score,
            "created": link.created.isoformat(),
            "context_snippet": link.context_snippet,
            "metadata": link.metadata
        }

    def _deserialize_link(self, data: dict[str, Any]) -> ConceptLink:
        """Convert JSON data to ConceptLink object."""
        return ConceptLink(
            link_id=data["link_id"],
            conversation_node_id=data["conversation_node_id"],
            conversation_tree_id=data["conversation_tree_id"],
            document_concept_id=data["document_concept_id"],
            document_id=data["document_id"],
            relationship_type=LinkRelationship(data["relationship_type"]),
            confidence_score=data["confidence_score"],
            created=datetime.fromisoformat(data["created"]),
            context_snippet=data.get("context_snippet", ""),
            metadata=data.get("metadata", {})
        )

    async def store_link(self, link: ConceptLink) -> bool:
        """
        Store a cross-reference link persistently.

        Args:
            link: ConceptLink to store

        Returns:
            True if stored successfully
        """
        try:
            # Store in memory
            self.links[link.link_id] = link

            # Update indices
            conv_id = link.conversation_tree_id
            concept_id = link.document_concept_id

            if conv_id not in self.conversation_to_concepts:
                self.conversation_to_concepts[conv_id] = []
            if concept_id not in self.conversation_to_concepts[conv_id]:
                self.conversation_to_concepts[conv_id].append(concept_id)

            if concept_id not in self.concept_to_conversations:
                self.concept_to_conversations[concept_id] = []
            if conv_id not in self.concept_to_conversations[concept_id]:
                self.concept_to_conversations[concept_id].append(conv_id)

            # Persist to disk
            link_file = self.links_dir / f"{link.link_id}.json"
            with open(link_file, 'w') as f:
                json.dump(self._serialize_link(link), f, indent=2)

            # Update index
            self._save_index()

            logger.info(f"Stored cross-reference link: {link.link_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store cross-reference link: {e}")
            return False

    async def store_links(self, links: list[ConceptLink]) -> int:
        """
        Store multiple cross-reference links.

        Args:
            links: List of ConceptLinks to store

        Returns:
            Number of successfully stored links
        """
        stored_count = 0
        for link in links:
            if await self.store_link(link):
                stored_count += 1

        return stored_count

    async def get_links_for_conversation(self, conversation_id: str) -> list[ConceptLink]:
        """
        Get all concept links for a conversation.

        Args:
            conversation_id: ID of conversation tree

        Returns:
            List of ConceptLinks for the conversation
        """
        concept_ids = self.conversation_to_concepts.get(conversation_id, [])

        links = []
        for link in self.links.values():
            if link.conversation_tree_id == conversation_id:
                links.append(link)

        return links

    async def get_links_for_concept(self, concept_id: str) -> list[ConceptLink]:
        """
        Get all conversation links for a concept.

        Args:
            concept_id: ID of document concept

        Returns:
            List of ConceptLinks for the concept
        """
        conversation_ids = self.concept_to_conversations.get(concept_id, [])

        links = []
        for link in self.links.values():
            if link.document_concept_id == concept_id:
                links.append(link)

        return links

    async def get_all_links(self,
                          relationship_type: LinkRelationship | None = None,
                          min_confidence: float = 0.0) -> list[ConceptLink]:
        """
        Get all cross-reference links with optional filtering.

        Args:
            relationship_type: Optional filter by relationship type
            min_confidence: Minimum confidence score (default: 0.0)

        Returns:
            List of ConceptLinks matching criteria
        """
        links = []

        for link in self.links.values():
            if link.confidence_score >= min_confidence:
                if relationship_type is None or link.relationship_type == relationship_type:
                    links.append(link)

        # Sort by confidence score descending
        links.sort(key=lambda x: x.confidence_score, reverse=True)

        return links

    async def remove_link(self, link_id: str) -> bool:
        """
        Remove a cross-reference link.

        Args:
            link_id: ID of link to remove

        Returns:
            True if removed successfully
        """
        try:
            if link_id not in self.links:
                return False

            link = self.links[link_id]

            # Remove from indices
            conv_id = link.conversation_tree_id
            concept_id = link.document_concept_id

            if conv_id in self.conversation_to_concepts:
                self.conversation_to_concepts[conv_id] = [
                    c for c in self.conversation_to_concepts[conv_id] if c != concept_id
                ]
                if not self.conversation_to_concepts[conv_id]:
                    del self.conversation_to_concepts[conv_id]

            if concept_id in self.concept_to_conversations:
                self.concept_to_conversations[concept_id] = [
                    c for c in self.concept_to_conversations[concept_id] if c != conv_id
                ]
                if not self.concept_to_conversations[concept_id]:
                    del self.concept_to_conversations[concept_id]

            # Remove from memory
            del self.links[link_id]

            # Remove from disk
            link_file = self.links_dir / f"{link_id}.json"
            if link_file.exists():
                link_file.unlink()

            # Update index
            self._save_index()

            logger.info(f"Removed cross-reference link: {link_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove cross-reference link: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about stored cross-references."""
        relationship_counts = {}
        for link in self.links.values():
            rel_type = link.relationship_type.value
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1

        return {
            "total_links": len(self.links),
            "total_conversations": len(self.conversation_to_concepts),
            "total_concepts": len(self.concept_to_conversations),
            "relationship_types": relationship_counts,
            "avg_confidence": sum(link.confidence_score for link in self.links.values()) / len(self.links) if self.links else 0
        }
