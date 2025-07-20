#!/usr/bin/env python3
"""
Simple Phase 2 Test - Blended Integration

Quick test to verify Phase 2 methods work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognitive_memory_engine.core.engine import CognitiveMemoryEngine
from cognitive_memory_engine.types import SystemConfig, RTMConfig, NeuralGainConfig, KnowledgeDomain

async def test_phase2_methods():
    """Test Phase 2 methods directly."""
    
    print("🧠 Testing Phase 2 Methods")
    print("=" * 40)
    
    # Initialize the engine
    config = SystemConfig(
        data_directory="/tmp/cme_test_simple",
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
        # Store a simple conversation
        conversation = [
            {"role": "user", "content": "What is SPL?"},
            {"role": "assistant", "content": "SPL stands for Structural Prompt Language."}
        ]
        
        conv_result = await engine.store_conversation(
            conversation=conversation,
            context={"topic": "SPL_test", "importance": 0.8}
        )
        conversation_id = conv_result["conversation_id"]
        print(f"✅ Stored conversation: {conversation_id}")
        
        # Store simple document knowledge
        doc_content = "SPL (Structural Prompt Language) provides formal semantic annotation for prompts."
        doc_result = await engine.store_document_knowledge(
            document_content=doc_content,
            root_concept="SPL",
            domain="ai_architecture"
        )
        print(f"✅ Stored document: {doc_result['document_id']}")
        
        # Test cross-reference linking
        print("\n🔗 Testing cross-reference linking...")
        cross_refs = await engine.link_conversation_to_knowledge(conversation_id)
        print(f"✅ Found {len(cross_refs)} cross-references")
        
        # Test unified query
        print("\n🔍 Testing unified query...")
        result = await engine.query_blended_knowledge("What is SPL?")
        print(f"✅ Query confidence: {result['confidence_score']:.2f}")
        print(f"✅ Formal knowledge: {len(result['formal_knowledge'])} concepts")
        print(f"✅ Conversation insights: {len(result['conversation_insights'].get('results', []))} results")
        print(f"✅ Cross-references: {len(result['cross_references'])}")
        
        # Test concept retrieval
        print("\n📚 Testing concept retrieval...")
        concept = await engine.get_concept("SPL")
        print(f"✅ Found concept: {concept['concept_name'] if concept else 'Not found'}")
        
        print("\n🎉 Phase 2 Methods Test: SUCCESS!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await engine.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_phase2_methods())
    sys.exit(0 if success else 1)
