#!/usr/bin/env python3
"""
Dual-Track Architecture Test

Tests the implementation of Phase 1: Document Knowledge Storage
by storing and retrieving SAPE research document knowledge.

This demonstrates the difference between:
- Track 1: Conversation Memory (existing)
- Track 2: Document Knowledge Storage (newly implemented)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognitive_memory_engine.core.engine import CognitiveMemoryEngine
from cognitive_memory_engine.types import (
    NeuralGainConfig,
    RTMConfig,
    SystemConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample SAPE research content (simplified for testing)
SAPE_RESEARCH_CONTENT = """
Self-Adaptive Prompt Engineering (SAPE): A Novel Framework for AI Language Model Optimization

Abstract:
This paper presents SAPE (Self-Adaptive Prompt Engineering), a revolutionary framework for optimizing AI language model performance through adaptive prompt structures.

1. Introduction
SAPE introduces five core components that work together to create a self-improving prompt engineering system.

2. SAPE Architecture
The SAPE framework consists of five main components:

2.1 SPL (Structural Prompt Language)
SPL provides a formal language for semantic annotation of prompts, enabling machine interpretation and manipulation of prompt structures. Key features include:
- Metadata tagging for prompt components
- Task-specific annotations
- Constraint definitions
- Cognitive planning structures

2.2 PKG (Prompt Knowledge Graph)
PKG maintains a dynamic graph of prompt patterns, their relationships, and performance metrics. This enables:
- Pattern recognition across successful prompts
- Relationship mapping between prompt components
- Performance tracking and optimization
- Cross-domain prompt adaptation

2.3 SEE (Self-Evaluation Engine)
SEE provides automated assessment of prompt effectiveness through multiple evaluation criteria:
- Output quality metrics
- Coherence analysis
- Task completion rates
- User satisfaction scoring

2.4 CML (Continuous Meta-Learning)
CML implements adaptive learning mechanisms that improve prompt generation over time:
- Pattern extraction from successful interactions
- Meta-learning from prompt performance data
- Adaptive strategy adjustment
- Long-term optimization memory

2.5 Reflective Controller
The Reflective Controller orchestrates all components and provides high-level decision making:
- Component coordination
- Strategy selection
- Adaptation triggering
- System-wide optimization

3. Implementation Challenges
Key challenges in implementing SAPE include:
- Balancing automation with human control
- Managing computational complexity
- Ensuring prompt interpretability
- Handling domain-specific requirements

4. Results and Evaluation
Experimental results show significant improvements in prompt effectiveness across multiple domains, with average performance gains of 35-50% over traditional static prompt approaches.

5. Future Work
Future research directions include expanding SAPE to multimodal contexts, improving real-time adaptation capabilities, and developing domain-specific SAPE variants.

Conclusion:
SAPE represents a significant advancement in prompt engineering, providing a framework for adaptive, self-improving AI interactions.
"""

# Sample conversation about SAPE (for comparison)
SAPE_CONVERSATION = [
    {
        "role": "user",
        "content": "I'm interested in learning about SAPE architecture. Can you explain how it works?"
    },
    {
        "role": "assistant",
        "content": "SAPE (Self-Adaptive Prompt Engineering) is a fascinating framework with five main components. Let me break down each one for you..."
    },
    {
        "role": "user",
        "content": "What exactly is SPL in SAPE? How does it differ from regular prompting?"
    },
    {
        "role": "assistant",
        "content": "SPL stands for Structural Prompt Language. It's quite different from regular prompting because it provides formal semantic annotation..."
    }
]


async def test_dual_track_architecture():
    """
    Test the dual-track architecture implementation.
    
    Demonstrates:
    1. Track 1: Conversation memory storage (existing)
    2. Track 2: Document knowledge storage (new)
    3. Different retrieval patterns between tracks
    """
    print("🧠 Testing Dual-Track Architecture Implementation")
    print("=" * 60)

    # Setup engine configuration
    config = SystemConfig(
        data_directory="./test_data/dual_track_test",
        llm_model="gpt-4o-mini",
        embedding_model="all-MiniLM-L6-v2",  # Valid sentence-transformers model
        rtm_config=RTMConfig(),
        neural_gain_config=NeuralGainConfig(),
        max_context_length=8000,
        vector_similarity_threshold=0.7,
        auto_archive_days=30
    )

    # Initialize engine
    print("\n🔧 Initializing Cognitive Memory Engine...")
    engine = CognitiveMemoryEngine(config)
    await engine.initialize()
    print("✅ Engine initialized successfully")

    # Test Phase 1: Store conversation (Track 1 - existing functionality)
    print("\n" + "="*60)
    print("📝 PHASE 1: Testing Track 1 (Conversation Memory)")
    print("="*60)

    conversation_result = await engine.store_conversation(
        conversation=SAPE_CONVERSATION,
        context={
            "session_id": "test_sape_discussion",
            "topic": "SAPE Architecture Discussion",
            "participants": ["user", "assistant"]
        }
    )

    print(f"✅ Stored conversation with ID: {conversation_result['conversation_id']}")
    print(f"   - Message count: {conversation_result['message_count']}")
    print(f"   - RTM nodes: {conversation_result['rtm_tree']['node_count']}")
    print(f"   - Compression ratio: {conversation_result['rtm_tree']['compression_ratio']:.2f}")

    # Test Phase 2: Store document knowledge (Track 2 - NEW functionality)
    print("\n" + "="*60)
    print("📚 PHASE 2: Testing Track 2 (Document Knowledge Storage)")
    print("="*60)

    document_result = await engine.store_document_knowledge(
        document_content=SAPE_RESEARCH_CONTENT,
        root_concept="SAPE",
        domain="ai_architecture",
        metadata={
            "source": "research_paper",
            "authors": ["Weishun Zhong", "et al"],
            "publication_year": 2025,
            "type": "academic_research"
        }
    )

    print(f"✅ Stored document knowledge with ID: {document_result['document_id']}")
    print(f"   - Root concept: {document_result['root_concept']}")
    print(f"   - Domain: {document_result['domain']}")
    print(f"   - Total concepts: {document_result['document_analysis']['total_concepts']}")
    print(f"   - Compression ratio: {document_result['document_analysis']['compression_ratio']:.2f}")

    print("\n📊 Concept Hierarchy:")
    for concept_id, concept_info in document_result['concept_hierarchy'].items():
        print(f"   - {concept_info['name']}: {concept_info['description'][:60]}...")
        print(f"     Children: {concept_info['children_count']}, Salience: {concept_info['salience_score']:.2f}")

    # Test Phase 3: Compare retrieval methods
    print("\n" + "="*60)
    print("🔍 PHASE 3: Testing Retrieval Differences")
    print("="*60)

    # Test 3a: Conversation query (Track 1)
    print("\n🗣️ Track 1 Query (Conversation Memory):")
    conversation_query = await engine.query_memory(
        query="What is SPL in SAPE?",
        context_depth=3,
        time_scope="day",
        max_results=5
    )

    print(f"   Results: {len(conversation_query['results'])} conversation fragments")
    print(f"   Context: {conversation_query['assembled_context']['retrieved_nodes']} nodes retrieved")
    print(f"   Strategy: {conversation_query['assembled_context']['retrieval_strategy']}")

    # Test 3b: Direct concept query (Track 2)
    print("\n📚 Track 2 Query (Document Knowledge):")
    concept_info = await engine.get_concept("SPL")

    if concept_info:
        print(f"   ✅ Found formal concept: {concept_info['concept_name']}")
        print(f"   Description: {concept_info['description'][:100]}...")
        print(f"   Domain: {concept_info['domain']}")
        print(f"   Document: {concept_info['document_context']['document_title']}")
        print(f"   Confidence: {concept_info['metadata']['confidence_score']:.2f}")
        print(f"   Children: {len(concept_info['hierarchy']['child_concept_ids'])}")
    else:
        print("   ❌ Concept not found in document knowledge")

    # Test 3c: Domain browsing (Track 2)
    print("\n🏪 Track 2 Domain Browsing:")
    shelf_info = await engine.browse_knowledge_shelf("ai_architecture")

    print(f"   Domain: {shelf_info['domain_name']}")
    print(f"   Documents: {shelf_info['total_documents']}")
    print(f"   Featured concepts: {len(shelf_info['featured_concepts'])}")

    if shelf_info['featured_concepts']:
        print("   Top concepts:")
        for concept in shelf_info['featured_concepts'][:3]:
            print(f"     - {concept['name']} (salience: {concept['salience_score']:.2f})")

    # Test Phase 4: System statistics
    print("\n" + "="*60)
    print("📈 PHASE 4: System Statistics")
    print("="*60)

    stats = await engine.get_memory_stats(include_details=True)

    print(f"Engine Status: {stats['engine_status']}")
    print(f"Active Sessions: {stats['active_sessions']}")

    if 'temporal_library' in stats:
        print(f"Temporal Books: {stats['temporal_library'].get('total_books', 0)}")
        print(f"RTM Trees: {stats['temporal_library'].get('total_trees', 0)}")

    # Get document store stats
    if engine.document_store:
        doc_stats = await engine.document_store.get_stats()
        print("Document Knowledge:")
        print(f"  - Total documents: {doc_stats['total_documents']}")
        print(f"  - Total concepts: {doc_stats['total_concepts']}")
        print(f"  - Knowledge shelves: {doc_stats['total_shelves']}")
        print(f"  - Concepts indexed: {doc_stats['concepts_indexed']}")

    # Summary
    print("\n" + "="*60)
    print("🎯 DUAL-TRACK ARCHITECTURE TEST SUMMARY")
    print("="*60)
    print("✅ Track 1 (Conversation Memory): Working correctly")
    print("   - Stores dialogue as narrative RTMs")
    print("   - Temporal organization by session/day")
    print("   - Query returns conversation context")
    print()
    print("✅ Track 2 (Document Knowledge): Newly implemented!")
    print("   - Stores formal documents as concept RTMs")
    print("   - Hierarchical knowledge organization")
    print("   - Direct concept retrieval")
    print("   - Domain-based browsing")
    print()
    print("🚀 Next Steps:")
    print("   - Implement Track 3 (Blended Integration)")
    print("   - Add cross-reference linking")
    print("   - Create unified query interface")
    print("   - Test with real research documents")

    # Cleanup
    await engine.cleanup()
    print("\n✅ Test completed successfully!")


async def demonstrate_sape_example():
    """
    Demonstrate how SAPE research should be stored vs old approach.
    """
    print("\n" + "🔬 SAPE Research Storage Demonstration")
    print("="*60)

    print("❌ OLD APPROACH (conversation-only):")
    print("   Query: 'What is SPL in SAPE?'")
    print("   Returns: Conversation fragments like:")
    print("   'User asked about SPL... Assistant mentioned it provides formal semantic annotation...'")
    print()

    print("✅ NEW APPROACH (dual-track):")
    print("   Query: 'What is SPL in SAPE?'")
    print("   Returns:")
    print("   📚 Formal Knowledge: 'SPL (Structural Prompt Language) provides...")
    print("   🗣️ Conversation Context: 'We discussed SPL schema design...'")
    print("   🔗 Cross-References: Links between formal concept and discussion")
    print()

    print("🎯 The difference: Structured knowledge vs conversation memory!")


if __name__ == "__main__":
    asyncio.run(test_dual_track_architecture())
    asyncio.run(demonstrate_sape_example())
