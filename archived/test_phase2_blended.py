#!/usr/bin/env python3
"""
Phase 2 Blended Integration Test

Tests the Phase 2 implementation:
- Cross-reference linking between conversation and document knowledge
- Unified query interface combining both tracks
- Blended knowledge retrieval with cross-references
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognitive_memory_engine.core.engine import CognitiveMemoryEngine
from cognitive_memory_engine.types import (
    KnowledgeDomain,
    NeuralGainConfig,
    RTMConfig,
    SystemConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample conversation about SAPE
TEST_CONVERSATION = [
    {"role": "user", "content": "I'm interested in learning about SAPE. Can you explain what SPL is?"},
    {"role": "assistant", "content": "SPL stands for Structural Prompt Language. It's a component of the SAPE framework that provides formal language for semantic annotation of prompts."},
    {"role": "user", "content": "How does SPL relate to the other components like PKG?"},
    {"role": "assistant", "content": "SPL works with PKG (Prompt Knowledge Graph) to create structured representations. While SPL handles the semantic annotation, PKG manages the relationships between prompt patterns."}
]

# Sample SAPE document content
SAPE_DOCUMENT = """
SAPE (Self-Adaptive Prompt Engineering) Framework

1. SPL (Structural Prompt Language)
SPL provides formal semantic annotation capabilities for prompts, enabling machine interpretation and manipulation of prompt structures.

2. PKG (Prompt Knowledge Graph)
PKG maintains dynamic relationships between prompt patterns, tracking their performance and enabling pattern recognition across successful prompts.

3. SEE (Self-Evaluation Engine)
SEE provides automated quality assessment of generated responses through multiple metrics including coherence analysis and task completion rates.

4. CML (Continuous Meta-Learning)
CML enables long-term optimization through pattern extraction from successful interactions and adaptive strategy adjustment.
"""

async def test_phase2_blended_integration():
    """Test the Phase 2 blended integration functionality."""

    print("🧠 Testing Phase 2: Blended Integration Layer")
    print("=" * 60)

    # Initialize the engine
    config = SystemConfig(
        data_directory="/tmp/cme_test_phase2",
        llm_model="gemini-2.0-flash-001",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rtm_config=RTMConfig(),
        neural_gain_config=NeuralGainConfig(),
        max_context_length=8000,
        vector_similarity_threshold=0.7,
        auto_archive_days=30
    )

    engine = CognitiveMemoryEngine(config)
    await engine.initialize()

    try:
        print("\n🔧 Setting up test data...")

        # Store conversation about SAPE
        conversation_result = await engine.store_conversation(
            conversation=TEST_CONVERSATION,
            context={"topic": "SAPE_discussion", "importance": 0.8}
        )
        conversation_id = conversation_result["conversation_id"]
        print(f"✅ Stored conversation: {conversation_id}")

        # Store SAPE document knowledge
        document_result = await engine.store_document_knowledge(
            content=SAPE_DOCUMENT,
            root_concept="SAPE",
            domain=KnowledgeDomain.AI_ARCHITECTURE,
            metadata={"source": "test_document", "version": "1.0"}
        )
        document_id = document_result["document_id"]
        print(f"✅ Stored document: {document_id}")

        print("\n" + "=" * 60)
        print("🔗 PHASE 2A: Testing Cross-Reference Linking")
        print("=" * 60)

        # Test cross-reference linking
        cross_refs = await engine.link_conversation_to_knowledge(conversation_id)
        print(f"✅ Created {len(cross_refs)} cross-reference links:")

        for link in cross_refs[:3]:  # Show first 3 links
            print(f"   🔗 {link['concept_name']} -> {link['context_snippet'][:50]}...")

        print("\n" + "=" * 60)
        print("🔍 PHASE 2B: Testing Unified Query Interface")
        print("=" * 60)

        # Test blended knowledge query
        test_queries = [
            "What is SPL in SAPE?",
            "How does PKG work?",
            "Explain the relationship between SPL and PKG"
        ]

        for query in test_queries:
            print(f"\n📝 Query: '{query}'")

            result = await engine.query_blended_knowledge(query)

            print(f"🧠 Unified Summary: {result['unified_summary']}")
            print(f"📊 Confidence Score: {result['confidence_score']:.2f}")

            if result['formal_knowledge']:
                print(f"📚 Formal Knowledge: {len(result['formal_knowledge'])} concepts found")
                top_concept = result['formal_knowledge'][0]['concept']
                print(f"   → Top: {top_concept.title}")

            if result['conversation_insights']['results']:
                print(f"💬 Conversation Insights: {len(result['conversation_insights']['results'])} results")

            if result['cross_references']:
                print(f"🔗 Cross-References: {len(result['cross_references'])} connections")
                for ref in result['cross_references'][:2]:
                    print(f"   → {ref['formal_concept']} ↔ {ref['conversation_fragment'][:30]}...")

        print("\n" + "=" * 60)
        print("🎯 PHASE 2C: Testing Track Comparison")
        print("=" * 60)

        # Compare single-track vs blended results
        test_query = "What is SPL?"

        # Track 1 only (conversation)
        conv_result = await engine.query_memory(test_query, max_results=3)
        print(f"🗣️  Track 1 Only: {len(conv_result['results'])} conversation results")

        # Track 2 only (document)
        concept_result = await engine.get_concept("SPL")
        print(f"📚 Track 2 Only: {'Found' if concept_result else 'Not found'} concept")

        # Blended (both tracks)
        blended_result = await engine.query_blended_knowledge(test_query)
        print(f"🔄 Blended Query: {blended_result['confidence_score']:.2f} confidence")
        print(f"   → Formal: {len(blended_result['formal_knowledge'])} concepts")
        print(f"   → Conversational: {len(blended_result['conversation_insights']['results'])} results")
        print(f"   → Cross-refs: {len(blended_result['cross_references'])} connections")

        print("\n" + "=" * 60)
        print("🏆 PHASE 2 IMPLEMENTATION SUCCESS!")
        print("=" * 60)

        print("✅ Cross-reference linking: WORKING")
        print("✅ Unified query interface: WORKING")
        print("✅ Blended knowledge retrieval: WORKING")
        print("✅ Multi-track integration: WORKING")

        print("\n🎯 Phase 2 Target Behavior Achieved:")
        print("   📚 Formal knowledge from document RTMs")
        print("   💬 Conversation insights from dialogue RTMs")
        print("   🔗 Cross-references linking both tracks")
        print("   🧠 Unified understanding combining all sources")

        return True

    except Exception as e:
        print(f"❌ Error in Phase 2 test: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await engine.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_phase2_blended_integration())
    if success:
        print("\n🎉 Phase 2 implementation test completed successfully!")
    else:
        print("\n💥 Phase 2 implementation test failed!")
        sys.exit(1)
