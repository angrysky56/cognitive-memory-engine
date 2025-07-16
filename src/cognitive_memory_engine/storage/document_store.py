"""
Document Knowledge Storage

Manages storage and retrieval of formal document knowledge structures,
implementing Track 2 of the dual-track architecture.

Stores DocumentRTM objects with hierarchical concept organization,
separate from conversation-based RTMs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import DocumentRTM, KnowledgeConcept, KnowledgeDomain, KnowledgeShelf

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    Storage and management for document-based knowledge structures.

    Handles:
    - Formal document RTM storage/retrieval
    - Concept hierarchy management
    - Knowledge shelf organization
    - Domain-based indexing
    """

    def __init__(self, storage_path: str):
        """
        Initialize document store.

        Args:
            storage_path: Base directory for document storage
        """
        self.storage_path = Path(storage_path)
        self.documents_dir = self.storage_path / "documents"
        self.shelves_dir = self.storage_path / "shelves"
        self.concepts_index_file = self.storage_path / "concepts_index.json"

        # In-memory caches
        self.documents: dict[str, DocumentRTM] = {}
        self.shelves: dict[str, KnowledgeShelf] = {}
        self.concepts_index: dict[str, str] = {}  # concept_name -> document_id

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)
        self.shelves_dir.mkdir(exist_ok=True)

        # Load existing concepts index
        if self.concepts_index_file.exists():
            try:
                with open(self.concepts_index_file) as f:
                    self.concepts_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load concepts index: {e}")

    async def store_document(self, document: DocumentRTM) -> bool:
        """
        Store a document RTM with full persistence.

        Args:
            document: DocumentRTM to store

        Returns:
            True if stored successfully
        """
        try:
            # Store in memory cache
            self.documents[document.doc_id] = document

            # Persist to disk
            doc_file = self.documents_dir / f"{document.doc_id}.json"

            # Convert to serializable format
            doc_data = {
                "doc_id": document.doc_id,
                "title": document.title,
                "root_concept": document.root_concept,
                "domain": document.domain.value,
                "root_concept_id": document.root_concept_id,
                "source_content": document.source_content,
                "source_metadata": document.source_metadata,
                "max_branching_factor": document.max_branching_factor,
                "max_depth": document.max_depth,
                "total_concepts": document.total_concepts,
                "compression_ratio": document.compression_ratio,
                "created": document.created.isoformat(),
                "last_accessed": document.last_accessed.isoformat(),
                "concepts": {}
            }

            # Serialize concepts
            for concept_id, concept in document.concepts.items():
                doc_data["concepts"][concept_id] = {
                    "concept_id": concept.concept_id,
                    "name": concept.name,
                    "description": concept.description,
                    "parent_concept_id": concept.parent_concept_id,
                    "child_concept_ids": concept.child_concept_ids,
                    "content": concept.content,
                    "structured_data": concept.structured_data,
                    "examples": concept.examples,
                    "base_embedding": concept.base_embedding,
                    "salience_score": concept.salience_score,
                    "related_concept_ids": concept.related_concept_ids,
                    "domain": concept.domain.value,
                    "tags": concept.tags,
                    "confidence_score": concept.confidence_score,
                    "created": concept.created.isoformat(),
                    "last_updated": concept.last_updated.isoformat()
                }

            # Write to file
            with open(doc_file, 'w') as f:
                json.dump(doc_data, f, indent=2)

            # Update concepts index
            for concept in document.concepts.values():
                self.concepts_index[concept.name.lower()] = document.doc_id

            await self._save_concepts_index()

            # Add to appropriate shelf
            await self._add_to_shelf(document)

            logger.info(f"Stored document {document.doc_id} with {len(document.concepts)} concepts")
            return True

        except Exception as e:
            logger.error(f"Failed to store document {document.doc_id}: {e}")
            return False

    async def get_document(self, doc_id: str) -> DocumentRTM | None:
        """
        Retrieve a document RTM by ID.

        Args:
            doc_id: Document ID to retrieve

        Returns:
            DocumentRTM if found, None otherwise
        """
        # Check memory cache first
        if doc_id in self.documents:
            self.documents[doc_id].last_accessed = datetime.now()
            return self.documents[doc_id]

        # Load from disk
        doc_file = self.documents_dir / f"{doc_id}.json"
        if not doc_file.exists():
            return None

        try:
            with open(doc_file) as f:
                doc_data = json.load(f)

            # Deserialize document
            document = DocumentRTM(
                doc_id=doc_data["doc_id"],
                title=doc_data["title"],
                root_concept=doc_data["root_concept"],
                domain=KnowledgeDomain(doc_data["domain"]),
                root_concept_id=doc_data["root_concept_id"],
                source_content=doc_data["source_content"],
                source_metadata=doc_data["source_metadata"],
                max_branching_factor=doc_data["max_branching_factor"],
                max_depth=doc_data["max_depth"],
                total_concepts=doc_data["total_concepts"],
                compression_ratio=doc_data["compression_ratio"],
                created=datetime.fromisoformat(doc_data["created"]),
                last_accessed=datetime.now()
            )

            # Deserialize concepts
            for concept_id, concept_data in doc_data["concepts"].items():
                concept = KnowledgeConcept(
                    concept_id=concept_data["concept_id"],
                    name=concept_data["name"],
                    description=concept_data["description"],
                    parent_concept_id=concept_data["parent_concept_id"],
                    child_concept_ids=concept_data["child_concept_ids"],
                    content=concept_data["content"],
                    structured_data=concept_data["structured_data"],
                    examples=concept_data["examples"],
                    base_embedding=concept_data["base_embedding"],
                    salience_score=concept_data["salience_score"],
                    related_concept_ids=concept_data["related_concept_ids"],
                    domain=KnowledgeDomain(concept_data["domain"]),
                    tags=concept_data["tags"],
                    confidence_score=concept_data["confidence_score"],
                    created=datetime.fromisoformat(concept_data["created"]),
                    last_updated=datetime.fromisoformat(concept_data["last_updated"])
                )
                document.concepts[concept_id] = concept

            # Cache in memory
            self.documents[doc_id] = document

            logger.info(f"Loaded document {doc_id} from disk")
            return document

        except Exception as e:
            logger.error(f"Failed to load document {doc_id}: {e}")
            return None

    async def get_concept_by_name(self, concept_name: str) -> tuple[DocumentRTM | None, KnowledgeConcept | None]:
        """
        Find a concept by name across all documents with fuzzy matching.

        Args:
            concept_name: Name of concept to find

        Returns:
            Tuple of (document, concept) if found, (None, None) otherwise
        """
        concept_lower = concept_name.lower().strip()

        # First try exact match in index
        doc_id = self.concepts_index.get(concept_lower)
        if doc_id:
            document = await self.get_document(doc_id)
            if document:
                for concept in document.concepts.values():
                    if concept.name.lower() == concept_lower:
                        return document, concept

        # If not found, try fuzzy matching (partial name matches)
        for indexed_name, doc_id in self.concepts_index.items():
            # Check if query is contained in the indexed name or vice versa
            if (concept_lower in indexed_name or
                indexed_name in concept_lower or
                # Handle abbreviated forms like "SPL" vs "SPL (Structural Prompt Language)"
                concept_lower == indexed_name.split('(')[0].strip().lower()):

                document = await self.get_document(doc_id)
                if document:
                    for concept in document.concepts.values():
                        concept_name_base = concept.name.split('(')[0].strip().lower()
                        if (concept.name.lower() == indexed_name or
                            concept_name_base == concept_lower or
                            concept_lower in concept.name.lower()):
                            logger.info(f"Found concept '{concept.name}' via fuzzy match for query '{concept_name}'")
                            return document, concept

        return None, None

    async def search_concepts(self, query: str, domain: KnowledgeDomain | None = None) -> list[tuple[DocumentRTM, KnowledgeConcept]]:
        """
        Search for concepts across documents.

        Args:
            query: Search query
            domain: Optional domain filter

        Returns:
            List of (document, concept) tuples matching query
        """
        results = []
        query_lower = query.lower()

        # Search through concepts index
        for concept_name, doc_id in self.concepts_index.items():
            if query_lower in concept_name:
                document = await self.get_document(doc_id)
                if document:
                    for concept in document.concepts.values():
                        if (concept.name.lower() == concept_name and
                            (domain is None or concept.domain == domain)):
                            results.append((document, concept))

        return results

    async def get_shelf(self, domain: KnowledgeDomain) -> KnowledgeShelf | None:
        """
        Get knowledge shelf for a domain.

        Args:
            domain: Knowledge domain

        Returns:
            KnowledgeShelf if exists, None otherwise
        """
        shelf_id = f"shelf_{domain.value}"

        # Check memory cache
        if shelf_id in self.shelves:
            return self.shelves[shelf_id]

        # Load from disk
        shelf_file = self.shelves_dir / f"{shelf_id}.json"
        if not shelf_file.exists():
            return None

        try:
            with open(shelf_file) as f:
                shelf_data = json.load(f)

            shelf = KnowledgeShelf(
                shelf_id=shelf_data["shelf_id"],
                domain=KnowledgeDomain(shelf_data["domain"]),
                name=shelf_data["name"],
                description=shelf_data["description"],
                document_ids=shelf_data["document_ids"],
                featured_concepts=shelf_data["featured_concepts"],
                subcategories=shelf_data["subcategories"],
                tags=shelf_data["tags"],
                created=datetime.fromisoformat(shelf_data["created"]),
                last_accessed=datetime.now()
            )

            # Cache in memory
            self.shelves[shelf_id] = shelf
            return shelf

        except Exception as e:
            logger.error(f"Failed to load shelf {shelf_id}: {e}")
            return None

    async def list_documents_by_domain(self, domain: KnowledgeDomain) -> list[DocumentRTM]:
        """
        List all documents in a knowledge domain.

        Args:
            domain: Knowledge domain

        Returns:
            List of documents in the domain
        """
        documents = []

        # Get shelf for domain
        shelf = await self.get_shelf(domain)
        if shelf:
            for doc_id in shelf.document_ids:
                document = await self.get_document(doc_id)
                if document:
                    documents.append(document)

        return documents

    async def get_all_documents(self) -> list[DocumentRTM]:
        """
        Get all documents across all domains.

        Returns:
            List of all documents in the store
        """
        documents = []

        # Get all domains
        for domain in KnowledgeDomain:
            domain_documents = await self.list_documents_by_domain(domain)
            documents.extend(domain_documents)

        return documents

    async def _add_to_shelf(self, document: DocumentRTM) -> None:
        """Add document to appropriate knowledge shelf."""
        shelf_id = f"shelf_{document.domain.value}"
        shelf = await self.get_shelf(document.domain)

        if not shelf:
            # Create new shelf
            shelf = KnowledgeShelf(
                shelf_id=shelf_id,
                domain=document.domain,
                name=f"{document.domain.value.replace('_', ' ').title()} Knowledge",
                description=f"Documents and concepts related to {document.domain.value}",
                document_ids=[document.doc_id],
                featured_concepts=[document.root_concept_id],
                created=datetime.now(),
                last_accessed=datetime.now()
            )
        else:
            # Add to existing shelf
            if document.doc_id not in shelf.document_ids:
                shelf.document_ids.append(document.doc_id)
            if document.root_concept_id not in shelf.featured_concepts:
                shelf.featured_concepts.append(document.root_concept_id)
            shelf.last_accessed = datetime.now()

        # Store shelf
        await self._store_shelf(shelf)

    async def _store_shelf(self, shelf: KnowledgeShelf) -> None:
        """Store knowledge shelf to disk."""
        try:
            shelf_file = self.shelves_dir / f"{shelf.shelf_id}.json"

            shelf_data = {
                "shelf_id": shelf.shelf_id,
                "domain": shelf.domain.value,
                "name": shelf.name,
                "description": shelf.description,
                "document_ids": shelf.document_ids,
                "featured_concepts": shelf.featured_concepts,
                "subcategories": shelf.subcategories,
                "tags": shelf.tags,
                "created": shelf.created.isoformat(),
                "last_accessed": shelf.last_accessed.isoformat()
            }

            with open(shelf_file, 'w') as f:
                json.dump(shelf_data, f, indent=2)

            # Cache in memory
            self.shelves[shelf.shelf_id] = shelf

        except Exception as e:
            logger.error(f"Failed to store shelf {shelf.shelf_id}: {e}")

    async def _save_concepts_index(self) -> None:
        """Save concepts index to disk."""
        try:
            with open(self.concepts_index_file, 'w') as f:
                json.dump(self.concepts_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save concepts index: {e}")

    async def get_stats(self) -> dict[str, Any]:
        """Get document store statistics."""
        total_documents = len(list(self.documents_dir.glob("*.json")))
        total_shelves = len(list(self.shelves_dir.glob("*.json")))

        # Count concepts across all documents
        total_concepts = 0
        for document in self.documents.values():
            total_concepts += len(document.concepts)

        return {
            "total_documents": total_documents,
            "total_shelves": total_shelves,
            "total_concepts": total_concepts,
            "loaded_documents": len(self.documents),
            "loaded_shelves": len(self.shelves),
            "concepts_indexed": len(self.concepts_index)
        }
