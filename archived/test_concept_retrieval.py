#!/usr/bin/env python3
"""
Quick Test: Concept Retrieval Improvement

Tests the improved fuzzy matching for concept retrieval
to verify that "SPL" finds "SPL (Structural Prompt Language)"
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognitive_memory_engine.core.engine import CognitiveMemoryEngine
from cognitive_memory_engine.types import SystemConfig, RTMConfig, NeuralGainConfig

async def test_concept_retrieval():
    """Test improved concept retrieval with fuzzy matching."""
    print("🔍 Testing Improved Concept Retrieval")
    print("=" * 50)
    
    # Setup minimal config
    config = SystemConfig(
        data_directory="./test_data/dual_track_test",  # Reuse previous test data
        llm_model="gpt-4o-mini",
        embedding_model="all-MiniLM-L6-v2",
        rtm_config=RTMConfig(),
        neural_gain_config=NeuralGainConfig(),
        max_context_length=8000,
        vector_similarity_threshold=0.7,
        auto_archive_days=30
    )
    
    # Initialize engine (should load existing data)
    print("🔧 Initializing engine with existing data...")
    engine = CognitiveMemoryEngine(config)
    await engine.initialize()
    
    # Test various concept retrieval patterns
    test_queries = [
        "SPL",  # Abbreviated form
        "spl",  # Lowercase
        "SPL (Structural Prompt Language)",  # Full name
        "Structural Prompt Language",  # Description part
        "PKG",  # Another abbreviation
        "SAPE",  # Root concept
        "Implementation Challenges",  # Multi-word concept
        "NonExistent",  # Should not be found
    ]
    
    print(f"\n📋 Testing {len(test_queries)} concept queries:")
    print("-" * 50)
    
    results = {}
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        try:
            concept_info = await engine.get_concept(query)
            
            if concept_info:
                results[query] = "✅ FOUND"
                print(f"   ✅ Found: {concept_info['concept_name']}")
                print(f"   📄 Document: {concept_info['document_context']['document_title']}")
                print(f"   📝 Description: {concept_info['description'][:80]}...")
                print(f"   🔗 Children: {len(concept_info['hierarchy']['child_concept_ids'])}")
            else:
                results[query] = "❌ NOT FOUND"
                print(f"   ❌ Not found")
                
        except Exception as e:
            results[query] = f"⚠️ ERROR: {e}"
            print(f"   ⚠️ Error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CONCEPT RETRIEVAL TEST RESULTS")
    print("=" * 50)
    
    found_count = sum(1 for result in results.values() if result == "✅ FOUND")
    not_found_count = sum(1 for result in results.values() if result == "❌ NOT FOUND")
    error_count = len(results) - found_count - not_found_count
    
    for query, result in results.items():
        print(f"{result:15} {query}")
    
    print(f"\n📈 Summary:")
    print(f"   ✅ Found: {found_count}/{len(test_queries)}")
    print(f"   ❌ Not Found: {not_found_count}/{len(test_queries)}")
    print(f"   ⚠️ Errors: {error_count}/{len(test_queries)}")
    
    # Key validation
    key_tests = {
        "SPL": results.get("SPL"),
        "PKG": results.get("PKG"),
        "SAPE": results.get("SAPE")
    }
    
    all_key_found = all(result == "✅ FOUND" for result in key_tests.values())
    
    print(f"\n🎯 Key Concept Validation:")
    for concept, result in key_tests.items():
        print(f"   {concept}: {result}")
    
    if all_key_found:
        print(f"\n🎉 SUCCESS: All key SAPE concepts can be retrieved!")
        print(f"   The fuzzy matching improvement is working correctly.")
    else:
        print(f"\n⚠️ ISSUE: Some key concepts still not found.")
        print(f"   May need further concept indexing improvements.")
    
    # Cleanup
    await engine.cleanup()
    
    return all_key_found

if __name__ == "__main__":
    success = asyncio.run(test_concept_retrieval())
    sys.exit(0 if success else 1)
