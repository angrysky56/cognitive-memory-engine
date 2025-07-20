#!/usr/bin/env python3
"""Test script to store the Intelligent Workspace narrative using the cognitive memory engine."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_memory_engine import CognitiveMemoryEngine

async def store_intelligent_workspace_narrative():
    """Store the Intelligent Workspace narrative in the cognitive memory engine."""
    
    # Initialize the engine
    engine = CognitiveMemoryEngine()
    
    # Define the conversation about the Intelligent Workspace
    conversation = [
        {
            "role": "system",
            "content": "Research documentation on the Intelligent Workspace unified memory architecture - a groundbreaking AI memory system that combines Random Tree Model hierarchical storage with Dynamic Working Memory management."
        },
        {
            "role": "assistant", 
            "content": "The Intelligent Workspace represents a unified AI memory model that synthesizes two complementary cognitive architectures into a powerful, human-like memory system. This architecture consists of two integrated components that work together seamlessly."
        },
        {
            "role": "system",
            "content": "Part 1: The Long-Term Store - Random Tree Architecture"
        },
        {
            "role": "assistant",
            "content": "The Long-Term Store uses the Random Tree Model as the foundational storage architecture. This is the brain's elegant solution for compressing and organizing vast amounts of narrative information, functioning like a comprehensive library. The AI's entire knowledge base - every document read, conversation held, and experience gained - is encoded into a massive forest of Random Trees. Each tree uses hierarchical summarization with high-level summaries at the root branching down to progressively more detailed sub-plots and individual events, with the finest details at the leaves. This structure enables highly efficient access to information at any level of abstraction, from high-level gist to verbatim details, without processing entire documents."
        },
        {
            "role": "system", 
            "content": "Part 2: The Active Workspace - Dynamic Memory Manager"
        },
        {
            "role": "assistant",
            "content": "The Active Workspace transforms the traditional fixed context window into an intelligent, dynamic processing system that mimics the brain's frontal cortex control mechanisms. When processing a task, the system performs: 1) Task Analysis - identifying key components in the prompt, 2) Intelligent Retrieval - sending targeted queries to the Random Tree store to fetch relevant summary nodes rather than raw text blocks, 3) Loading the Workspace - populating with highly-relevant, pre-summarized information, 4) Dynamic Prioritization - using neural gain mechanisms to assign highest importance to new goals while actively suppressing irrelevant details to prevent proactive interference, and 5) Synthesis and Reasoning - using prioritized focus to guide thoughts and integrate high-gain concepts for novel solutions."
        },
        {
            "role": "system",
            "content": "Breakthrough Capabilities and Implementation"
        },
        {
            "role": "assistant", 
            "content": "This unified architecture creates capabilities that surpass traditional AI limitations: immunity to interference (maintains long complex conversations without older information disrupting new understanding), deep multi-step reasoning (tackles problems requiring integration of diverse high-level knowledge), true contextual understanding (recognizes that information meaning changes based on current goals), and massive computational efficiency (processes small, highly-relevant, prioritized information sets rather than large passive text blocks). The system essentially creates both a vast, well-organized knowledge library and a brilliant, focused librarian that knows exactly what to retrieve and arrange for any given task. Technical implementation involves using sentence transformers for embeddings, ChromaDB for vector storage, NetworkX for Random Tree graph structures, and neural gain magnitude encoding for salience-weighted retrieval."
        }
    ]
    
    # Context for the conversation
    context = {
        "topic": "Intelligent Workspace Unified Memory Architecture",
        "importance": 0.95,
        "domain": "AI Memory Systems", 
        "research_phase": "architecture_design",
        "participants": ["Research Team", "AI System"]
    }
    
    try:
        # Store the conversation
        print("Storing Intelligent Workspace narrative...")
        result = await engine.store_conversation(conversation, context)
        print(f"✅ Narrative stored successfully!")
        print(f"Result: {result}")
        
        # Get memory stats
        print("\nMemory statistics:")
        stats = await engine.get_memory_stats()
        print(f"Stats: {stats}")
        
        # Test retrieval
        print("\nTesting retrieval...")
        response = await engine.query_memory("What is the Intelligent Workspace architecture?")
        print(f"Query response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(store_intelligent_workspace_narrative())
