import asyncio
import sys
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognitive_memory_engine.storage.document_store import DocumentStore
from cognitive_memory_engine.core.engine import CognitiveMemoryEngine
from cognitive_memory_engine.types import KnowledgeDomain, DocumentRTM, KnowledgeShelf


async def test_document_store_pagination():
    print("Testing DocumentStore pagination...")

    # Mock dependencies
    config = MagicMock()
    config.cme_data_dir = Path("/tmp/cme_test_data")

    store = DocumentStore(config)

    # Mock get_shelf to return a shelf with many document IDs
    mock_shelf = MagicMock(spec=KnowledgeShelf)
    # Create 50 dummy document IDs
    mock_shelf.document_ids = [f"doc_{i}" for i in range(50)]
    store.get_shelf = AsyncMock(return_value=mock_shelf)

    # Mock get_document to return a dummy document
    async def mock_get_document(doc_id):
        doc = MagicMock(spec=DocumentRTM)
        doc.doc_id = doc_id
        return doc

    store.get_document = mock_get_document

    # Test case 1: Default limit (should be 100, so all 50)
    docs = await store.list_documents_by_domain(KnowledgeDomain.AI_ARCHITECTURE)
    print(f"Test 1 (Default): Got {len(docs)} documents. Expected 50.")
    assert len(docs) == 50

    # Test case 2: Limit 10
    docs = await store.list_documents_by_domain(KnowledgeDomain.AI_ARCHITECTURE, limit=10)
    print(f"Test 2 (Limit 10): Got {len(docs)} documents. Expected 10.")
    assert len(docs) == 10
    assert docs[0].doc_id == "doc_0"
    assert docs[9].doc_id == "doc_9"

    # Test case 3: Offset 10, Limit 10
    docs = await store.list_documents_by_domain(KnowledgeDomain.AI_ARCHITECTURE, limit=10, offset=10)
    print(f"Test 3 (Offset 10, Limit 10): Got {len(docs)} documents. Expected 10.")
    assert len(docs) == 10
    assert docs[0].doc_id == "doc_10"
    assert docs[9].doc_id == "doc_19"

    # Test case 4: Offset 45, Limit 10 (should return last 5)
    docs = await store.list_documents_by_domain(KnowledgeDomain.AI_ARCHITECTURE, limit=10, offset=45)
    print(f"Test 4 (Offset 45, Limit 10): Got {len(docs)} documents. Expected 5.")
    assert len(docs) == 5
    assert docs[0].doc_id == "doc_45"
    assert docs[4].doc_id == "doc_49"

    print("DocumentStore pagination passed!")


async def test_engine_pagination():
    print("\nTesting CognitiveMemoryEngine pagination...")

    # Mock engine and document store
    config = MagicMock()
    config.cme_data_dir = Path("/tmp/cme_test_data")

    # We need to mock the engine's internal components initialization to avoid real IO
    # Using patch to bypass __init__ logic completely
    with patch("cognitive_memory_engine.core.engine.CognitiveMemoryEngine.__init__", return_value=None):
        engine = CognitiveMemoryEngine(config)
        # Manually set attributes that __init__ would have set
        engine.config = config
        engine.initialized = True
        engine.document_store = AsyncMock(spec=DocumentStore)

        # Mock _ensure_component to do nothing
        engine._ensure_component = MagicMock()

        # Mock list_documents_by_domain to return a list of docs based on limit
        async def mock_list_docs(domain, limit=100, offset=0):
            return [MagicMock() for _ in range(limit)]

        engine.document_store.list_documents_by_domain = mock_list_docs

        # Mock get_shelf to return a shelf with total count
        mock_shelf = MagicMock()
        mock_shelf.document_ids = ["doc"] * 100  # Total 100 docs
        engine.document_store.get_shelf = AsyncMock(return_value=mock_shelf)

        # Test browse_knowledge_shelf
        result = await engine.browse_knowledge_shelf("ai_architecture", limit=5, offset=10)

        print(f"Result keys: {result.keys()}")
        print(f"Pagination info: {result.get('pagination')}")

        assert result["returned_documents"] == 5
        assert result["total_documents"] == 100
        assert result["pagination"]["limit"] == 5
        assert result["pagination"]["offset"] == 10
        assert result["pagination"]["has_more"] is True

        print("CognitiveMemoryEngine pagination passed!")


if __name__ == "__main__":
    asyncio.run(test_document_store_pagination())
    asyncio.run(test_engine_pagination())
