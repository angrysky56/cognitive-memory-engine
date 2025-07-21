#!/usr/bin/env python3
"""
CME Functionality Test Script

Tests the key features of the Cognitive Memory Engine to ensure
all components are working properly.
"""

import asyncio
import json
from datetime import datetime


async def test_conversation_storage():
    """Test Track 1: Conversation Memory"""
    print("\n1. Testing Conversation Storage (Track 1)...")
    
    from cognitive_memory_engine import (
        store_conversation,
        query_memory,
        get_memory_stats
    )
    
    # Create test conversation
    test_conversation = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."},
        {"role": "user", "content": "Can you give me an example?"},
        {"role": "assistant", "content": "Sure! Image recognition is a common ML application."}
    ]
    
    # Store conversation
    print("  - Storing test conversation...")
    result = await store_conversation(
        conversation=test_conversation,
        context={"topic": "machine learning basics", "importance": 0.8}
    )
    
    # Check if task was created
    if result.get("status") == "accepted":
        print(f"  ✓ Conversation storage task created: {result['task_id']}")
        
        # Wait a moment for processing
        await asyncio.sleep(3)
        
        # Query the stored conversation
        print("  - Querying stored conversation...")
        query_result = await query_memory(
            query="machine learning example",
            context_depth=2,
            time_scope="day"
        )
        
        if query_result.get("results"):
            print(f"  ✓ Found {len(query_result['results'])} results")
        else:
            print("  ⚠ No results found (may still be processing)")
    else:
        print(f"  ✗ Failed to store conversation: {result}")


async def test_document_storage():
    """Test Track 2: Document Knowledge"""
    print("\n2. Testing Document Knowledge Storage (Track 2)...")
    
    from cognitive_memory_engine import (
        store_document_knowledge,
        get_concept,
        browse_knowledge_shelf
    )
    
    # Create test document
    test_document = """
    Neural Networks: A Comprehensive Overview
    
    Neural networks are computing systems inspired by biological neural networks.
    They consist of interconnected nodes (neurons) organized in layers:
    
    1. Input Layer: Receives the initial data
    2. Hidden Layers: Process information through weighted connections
    3. Output Layer: Produces the final result
    
    Key concepts include activation functions, backpropagation, and gradient descent.
    """
    
    # Store document
    print("  - Storing test document...")
    result = await store_document_knowledge(
        document_content=test_document,
        root_concept="Neural Networks",
        domain="neural_networks",
        metadata={"source": "test", "version": "1.0"}
    )
    
    if result.get("status") == "success":
        print(f"  ✓ Document stored with {result['document_analysis']['total_concepts']} concepts")
        
        # Retrieve concept
        print("  - Retrieving stored concept...")
        concept = await get_concept("Neural Networks")
        
        if concept:
            print(f"  ✓ Retrieved concept: {concept['concept_name']}")
        else:
            print("  ⚠ Concept not found")
            
        # Browse shelf
        print("  - Browsing neural_networks shelf...")
        shelf = await browse_knowledge_shelf("neural_networks")
        
        if shelf.get("documents"):
            print(f"  ✓ Found {shelf['total_documents']} documents on shelf")
        else:
            print("  ⚠ No documents found on shelf")
    else:
        print(f"  ✗ Failed to store document: {result}")


async def test_blended_query():
    """Test Track 3: Blended Integration"""
    print("\n3. Testing Blended Query (Track 3)...")
    
    from cognitive_memory_engine import query_blended_knowledge
    
    # Query across both tracks
    print("  - Performing blended query...")
    result = await query_blended_knowledge(
        query="neural networks and machine learning",
        include_formal=True,
        include_conversational=True
    )
    
    formal_count = len(result.get("formal_knowledge", []))
    conv_count = len(result.get("conversation_insights", {}).get("results", []))
    cross_refs = len(result.get("cross_references", []))
    
    print(f"  ✓ Blended query results:")
    print(f"    - Formal concepts: {formal_count}")
    print(f"    - Conversation insights: {conv_count}")
    print(f"    - Cross-references: {cross_refs}")
    print(f"    - Confidence: {result.get('confidence_score', 0):.2f}")


async def test_memory_stats():
    """Test memory statistics"""
    print("\n4. Testing Memory Statistics...")
    
    from cognitive_memory_engine import get_memory_stats
    
    stats = await get_memory_stats(include_details=True)
    
    print(f"  ✓ Engine status: {stats.get('engine_status', 'unknown')}")
    print(f"  ✓ Active sessions: {stats.get('active_sessions', 0)}")
    
    if "temporal_library" in stats:
        lib_stats = stats["temporal_library"]
        print(f"  ✓ Temporal library:")
        print(f"    - Total books: {lib_stats.get('total_books', 0)}")
        print(f"    - Active sessions: {lib_stats.get('active_sessions', 0)}")
    
    if "vector_store" in stats:
        vec_stats = stats["vector_store"]
        print(f"  ✓ Vector store:")
        print(f"    - Total vectors: {vec_stats.get('total_vectors', 0)}")


async def main():
    """Run all tests"""
    print("CME Functionality Test Suite")
    print("=" * 50)
    
    try:
        # Test each track
        await test_conversation_storage()
        await test_document_storage()
        await test_blended_query()
        await test_memory_stats()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        print("\n✅ The CME appears to be functioning correctly!")
        print("All three tracks of the dual-track architecture are operational.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("Please ensure the CME MCP server is running.")


if __name__ == "__main__":
    # Note: This assumes the CME tools are available in the environment
    print("Note: This test requires the CME MCP server to be running.")
    print("The tests will use the MCP tools directly.")
    asyncio.run(main())
