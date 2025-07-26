#!/usr/bin/env python3
"""
Quick Import Test for Cognitive Memory Engine

Tests that all imports work correctly before running the full demo.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all core imports"""
    print("🧪 Testing Cognitive Memory Engine imports...")

    try:
        # Test main engine import
        print("  📦 Importing main engine...")
        from cognitive_memory_engine import CognitiveMemoryEngine
        print("  ✅ CognitiveMemoryEngine imported")

        # Test type imports
        print("  📦 Importing types...")
        from cognitive_memory_engine import (
            ConversationTurn,
            GeneratedResponse,
            LibraryShelf,
            MemoryQuery,
            RetrievalContext,
            RTMNode,
            RTMTree,
            TemporalBook,
        )
        print("  ✅ All types imported")

        # Test component imports
        print("  📦 Importing components...")
        from cognitive_memory_engine.comprehension import (
            NarrativeTreeBuilder,
            TemporalOrganizer,
        )
        from cognitive_memory_engine.production import ResponseGenerator
        from cognitive_memory_engine.storage import RTMGraphStore, TemporalLibrary
        from cognitive_memory_engine.workspace import ContextAssembler, VectorManager
        print("  ✅ All components imported")

        # Test optional dependencies
        print("  📦 Testing optional dependencies...")

        try:
            import ollama
            print("  ✅ Ollama available")
        except ImportError:
            print("  ⚠️  Ollama not installed (pip install ollama)")

        try:
            import chromadb
            print("  ✅ ChromaDB available")
        except ImportError:
            print("  ⚠️  ChromaDB not installed (pip install chromadb)")

        try:
            from sentence_transformers import SentenceTransformer
            print("  ✅ SentenceTransformers available")
        except ImportError:
            print("  ⚠️  SentenceTransformers not installed (pip install sentence-transformers)")

        try:
            import networkx
            print("  ✅ NetworkX available")
        except ImportError:
            print("  ⚠️  NetworkX not installed (pip install networkx)")

        print("\n🎉 Import test completed successfully!")
        print("\nTo run the full demo:")
        print("  python working_demo.py")
        print("\nTo test individual components:")
        print("  python working_demo.py --components")

        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nTroubleshooting:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check Python path")
        print("3. Ensure all files are present")
        return False

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic object creation without external dependencies"""
    print("\n🔧 Testing basic functionality...")

    try:
        from datetime import datetime

        from cognitive_memory_engine.types import (
            ConversationTurn,
            MemoryQuery,
            RTMNode,
            TemporalBook,
        )

        # Test ConversationTurn creation
        turn = ConversationTurn(
            role="user",
            content="Test message",
            timestamp=datetime.now()
        )
        print(f"  ✅ ConversationTurn: {turn.role} - {turn.content[:20]}...")

        # Test MemoryQuery creation
        query = MemoryQuery(
            query="Test query",
            max_context_depth=3,
            temporal_scope="day"
        )
        print(f"  ✅ MemoryQuery: {query.query}")

        # Test RTMNode creation
        from cognitive_memory_engine.types import NodeType, TemporalScale
        node = RTMNode(
            content="Test content",
            summary="Test summary",
            node_type=NodeType.LEAF,
            temporal_scale=TemporalScale.DAY
        )
        print(f"  ✅ RTMNode: {node.node_type.value} - {node.content[:20]}...")

        # Test TemporalBook creation
        from cognitive_memory_engine.types import ShelfCategory
        book = TemporalBook(
            title="Test Book",
            description="Test Description",
            temporal_scale=TemporalScale.DAY,
            shelf_category=ShelfCategory.ACTIVE
        )
        print(f"  ✅ TemporalBook: {book.title} ({book.shelf_category.value})")

        print("  🎉 Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧠 Cognitive Memory Engine - Import & Basic Tests")
    print("=" * 55)

    # Test imports
    import_success = test_imports()

    if import_success:
        # Test basic functionality
        basic_success = test_basic_functionality()

        if basic_success:
            print("\n✅ All tests passed! System ready for demo.")
        else:
            print("\n⚠️  Basic functionality issues detected.")
    else:
        print("\n❌ Import issues must be resolved first.")

    print("\n" + "=" * 55)

if __name__ == "__main__":
    main()
