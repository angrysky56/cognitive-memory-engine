#!/usr/bin/env python3
"""
Basic Usage Example for Cognitive Memory Engine

Demonstrates the key concepts:
1. RTM narrative tree building from conversations
2. Temporal organization into books and shelves
3. Neural gain weighted vector storage
4. Intelligent memory retrieval for response generation

This example shows how the asymmetric architecture works:
- Comprehension Module: Long-timescale integration and storage
- Production Module: Short-timescale generation from prioritized context
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# These imports will work once we complete the implementation
# from cognitive_memory_engine import CognitiveMemoryEngine, ConversationTurn, MemoryQuery

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_step(step: str, description: str):
    """Print a formatted step"""
    print(f"\n🔄 {step}")
    print(f"   {description}")

async def demonstrate_rtm_narrative_trees():
    """Demonstrate RTM tree building from conversations"""
    print_header("Random Tree Model (RTM) Narrative Processing")

    # Sample conversation about a project
    sample_conversation = [
        {
            "role": "user",
            "content": "I need help planning the Phoenix project timeline for Q3 launch"
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help with the Phoenix project timeline. What are the main components that need to be completed before the Q3 launch?"
        },
        {
            "role": "user",
            "content": "We have three main tracks: frontend development, backend API integration, and user testing. The integration is taking longer than expected."
        },
        {
            "role": "assistant",
            "content": "Integration delays can impact the whole timeline. What specifically is causing the backend integration delays? Is it API compatibility, data migration, or performance issues?"
        },
        {
            "role": "user",
            "content": "It's mainly API compatibility. The third-party service changed their authentication method and we need to refactor our connection logic."
        },
        {
            "role": "assistant",
            "content": "API compatibility changes can be tricky. For your Q3 timeline, I'd recommend: 1) Parallel work on frontend while fixing API issues, 2) Set up a staging environment to test the new auth flow, 3) Create a buffer week before Q3 launch for integration testing."
        }
    ]

    print_step("Step 1: Conversation Segmentation",
               "Breaking conversation into meaningful clauses using LLM")

    # Simulate the segmentation process
    simulated_clauses = [
        "user: Need help planning Phoenix project timeline for Q3 launch",
        "assistant: Happy to help with Phoenix project timeline planning",
        "user: Three main tracks - frontend, backend API, user testing",
        "user: Integration taking longer than expected",
        "assistant: Integration delays can impact whole timeline",
        "assistant: What's causing backend integration delays specifically",
        "user: API compatibility issues with third-party service",
        "user: Third-party changed authentication method",
        "user: Need to refactor connection logic",
        "assistant: API compatibility changes are tricky",
        "assistant: Recommend parallel frontend work",
        "assistant: Set up staging environment for auth flow testing",
        "assistant: Create buffer week before Q3 launch"
    ]

    print(f"   Extracted {len(simulated_clauses)} semantic clauses:")
    for i, clause in enumerate(simulated_clauses[:5], 1):
        print(f"     {i}. {clause}")
    print(f"     ... and {len(simulated_clauses)-5} more")

    print_step("Step 2: RTM Tree Building",
               "Hierarchical summarization with branching factor K=4")

    # Simulate the RTM tree structure
    tree_structure = {
        "root": {
            "summary": "Phoenix project Q3 timeline planning with API integration challenges",
            "children": [
                {
                    "summary": "Project scope: frontend, backend, testing tracks",
                    "children": ["user: Three main tracks", "user: Integration delays"]
                },
                {
                    "summary": "API compatibility issues with third-party auth changes",
                    "children": ["user: API compatibility", "user: Auth method changed"]
                },
                {
                    "summary": "Recommendations for timeline management",
                    "children": ["assistant: Parallel work", "assistant: Staging environment", "assistant: Buffer week"]
                }
            ]
        }
    }

    print("   RTM Tree Structure (K=4 branching):")
    print("   ROOT: Phoenix project Q3 timeline planning with API integration challenges")
    print("   ├── Project scope: frontend, backend, testing tracks")
    print("   │   ├── Three main tracks identified")
    print("   │   └── Integration delays noted")
    print("   ├── API compatibility issues with third-party auth changes")
    print("   │   ├── API compatibility problems")
    print("   │   └── Authentication method changed")
    print("   └── Recommendations for timeline management")
    print("       ├── Parallel work suggested")
    print("       ├── Staging environment setup")
    print("       └── Buffer week recommended")

    print(f"   📊 Compression: {len(simulated_clauses)} clauses → 7 nodes (ratio: {len(simulated_clauses)/7:.1f}x)")

async def demonstrate_temporal_organization():
    """Demonstrate temporal books and shelves organization"""
    print_header("Temporal Books & Shelves Organization")

    print_step("Step 1: Temporal Scaling",
               "Organizing conversations across time scales")

    # Simulate temporal organization
    temporal_scales = {
        "minute": "Individual conversation turns",
        "hour": "Complete conversation sessions",
        "day": "Daily project discussions",
        "week": "Weekly project progress",
        "month": "Monthly project phases",
        "year": "Annual project cycles"
    }

    for scale, description in temporal_scales.items():
        compression = {"minute": 1.0, "hour": 1.2, "day": 2.0, "week": 4.0, "month": 8.0, "year": 16.0}[scale]
        print(f"   📅 {scale.upper()}: {description} (compression: {compression}x)")

    print_step("Step 2: Shelf Categories",
               "Auto-organizing books by temporal relevance")

    shelf_categories = {
        "Active": "Currently developing conversations (< 1 day old)",
        "Recent": "Recently completed discussions (1-7 days old)",
        "Reference": "Persistent themes and important decisions",
        "Archived": "Older conversations with compressed summaries (> 30 days)"
    }

    for category, description in shelf_categories.items():
        print(f"   📚 {category}: {description}")

    print_step("Step 3: Temporal Book Example",
               "How our Phoenix conversation gets organized")

    print("   📖 Book: 'Project Planning - Day 2025-07-13'")
    print("      Category: Active")
    print("      Temporal Scale: Day")
    print("      RTM Trees: [Phoenix_Timeline_Discussion]")
    print("      Persistent Themes: ['API Integration', 'Q3 Launch', 'Timeline Management']")
    print("      Compression Ratio: 2.0x (daily level)")

async def demonstrate_neural_gain_mechanism():
    """Demonstrate neural gain salience weighting"""
    print_header("Neural Gain Mechanism for Salience Weighting")

    print_step("Step 1: Base Embeddings",
               "Generate normalized semantic vectors")

    # Simulate embedding process
    sample_nodes = [
        {"content": "API compatibility issues", "type": "root", "depth": 0},
        {"content": "Timeline management recommendations", "type": "summary", "depth": 1},
        {"content": "Need buffer week before Q3 launch", "type": "leaf", "depth": 2}
    ]

    print("   🔢 Base embeddings (normalized to unit length):")
    for node in sample_nodes:
        print(f"     '{node['content'][:30]}...' → [0.15, -0.23, 0.41, ...] (384-dim)")

    print_step("Step 2: Salience Calculation",
               "Multiple factors determine neural gain")

    salience_factors = {
        "Temporal Recency": "Recent content gets higher salience (exponential decay)",
        "Hierarchical Depth": "Root/summary nodes more important than leaves",
        "Content Richness": "Longer, more detailed content gets boost",
        "Temporal Scale": "Day/hour scale more important than year/minute"
    }

    for factor, description in salience_factors.items():
        print(f"   ⚡ {factor}: {description}")

    print_step("Step 3: Neural Gain Application",
               "Vector magnitude encodes priority")

    # Simulate neural gain calculation
    sample_salience = [
        {"content": "API compatibility issues", "salience": 2.3, "reason": "root node + recent + high relevance"},
        {"content": "Timeline recommendations", "salience": 1.8, "reason": "summary node + actionable advice"},
        {"content": "Buffer week suggestion", "salience": 1.2, "reason": "leaf node but important detail"}
    ]

    print("   🎯 Final weighted embeddings:")
    for item in sample_salience:
        print(f"     '{item['content'][:25]}...'")
        print(f"       Salience: {item['salience']} ({item['reason']})")
        print(f"       Vector: [0.15×{item['salience']}, -0.23×{item['salience']}, 0.41×{item['salience']}, ...]")
        print()

async def demonstrate_asymmetric_processing():
    """Demonstrate comprehension vs production modules"""
    print_header("Asymmetric Neural Architecture")

    print_step("Comprehension Module (Long-timescale)",
               "Builds and maintains rich narrative context")

    comprehension_tasks = [
        "🔍 Parse conversation into RTM narrative tree",
        "📚 Organize into temporal books and shelves",
        "🎯 Generate neural gain weighted embeddings",
        "🧠 Update social context and trust scores",
        "🔗 Link to existing knowledge and themes",
        "💾 Commit to persistent memory stores"
    ]

    for task in comprehension_tasks:
        print(f"   {task}")

    print_step("Production Module (Short-timescale)",
               "Generates responses from prioritized context")

    production_tasks = [
        "❓ Receive structured query from user",
        "🎯 Retrieve high-salience context using neural gain",
        "📋 Assemble prioritized context window",
        "🎭 Apply social governance (trust, formality)",
        "💬 Generate response using local LLM",
        "📊 Track prediction errors for learning"
    ]

    for task in production_tasks:
        print(f"   {task}")

    print_step("Example Query Processing",
               "How a question about Phoenix project gets answered")

    query_flow = [
        "🔍 Query: 'What was the main blocker for Phoenix project?'",
        "🎯 Vector search retrieves: API compatibility issues (salience: 2.3)",
        "🌳 RTM traversal finds: Authentication method changes",
        "📚 Temporal context: From today's active project discussion",
        "🎭 Social context: High trust, technical formality",
        "💬 Generated response: 'The main blocker was API compatibility issues...'",
        "⏱️  Processing time: ~200ms (local inference)"
    ]

    for step in query_flow:
        print(f"   {step}")

async def demonstrate_complete_workflow():
    """Show the complete workflow from conversation to query"""
    print_header("Complete Cognitive Memory Engine Workflow")

    print_step("Initialization",
               "Starting a new session")

    # This would be actual code once implemented:
    """
    cme = CognitiveMemoryEngine(
        data_dir="./demo_data",
        ollama_model="qwen2.5:7b"
    )
    session_id = await cme.start_session("demo_session")
    """

    print("   🧠 Cognitive Memory Engine initialized")
    print("   📁 Data directory: ./demo_data")
    print("   🤖 Local LLM: qwen2.5:7b via Ollama")
    print("   🆔 Session: demo_session_20250713")

    print_step("Memory Ingestion (Comprehension)",
               "Processing Phoenix project conversation")

    # Simulated ingestion result
    ingestion_result = {
        "rtm_tree_id": "tree_phoenix_timeline",
        "temporal_book_id": "book_day_20250713",
        "nodes_created": 7,
        "compression_ratio": 1.9,
        "vectors_stored": 9
    }

    print(f"   ✅ RTM tree built: {ingestion_result['nodes_created']} nodes")
    print(f"   📊 Compression ratio: {ingestion_result['compression_ratio']}x")
    print(f"   🎯 Neural gain vectors stored: {ingestion_result['vectors_stored']}")
    print("   📚 Organized into temporal book: Daily_ProjectPlanning")

    print_step("Memory Query (Production)",
               "Answering question about project blockers")

    # Simulated query result
    query_result = {
        "response": "The main blocker for the Phoenix project is API compatibility issues. The third-party service changed their authentication method, requiring refactoring of the connection logic. This is impacting the Q3 launch timeline.",
        "context_nodes": 4,
        "max_salience": 2.3,
        "temporal_books_accessed": ["book_day_20250713"],
        "generation_time_ms": 180
    }

    print(f"   🎯 High-salience context retrieved: {query_result['context_nodes']} nodes")
    print(f"   📊 Max salience score: {query_result['max_salience']}")
    print(f"   ⏱️  Response generation: {query_result['generation_time_ms']}ms")
    print(f"   💬 Response: {query_result['response'][:100]}...")

async def main():
    """Run all demonstrations"""
    print("🧠 Cognitive Memory Engine - Demonstration")
    print("Showcasing neuroscience-inspired AI memory architecture")

    await demonstrate_rtm_narrative_trees()
    await demonstrate_temporal_organization()
    await demonstrate_neural_gain_mechanism()
    await demonstrate_asymmetric_processing()
    await demonstrate_complete_workflow()

    print_header("Summary: Key Innovations")

    innovations = [
        "🌳 RTM Narrative Trees: Hierarchical compression like human memory",
        "📚 Temporal Organization: Books & shelves across time scales",
        "🎯 Neural Gain: Vector magnitude encodes salience/priority",
        "🧠 Asymmetric Architecture: Separate comprehension/production modules",
        "🤖 Local-First: Runs entirely on your hardware via Ollama",
        "💾 Persistent Memory: True long-term conversation memory",
        "🔗 Knowledge Continuity: Themes persist across conversations"
    ]

    for innovation in innovations:
        print(f"   {innovation}")

    print("\n🎉 This demonstrates how the Cognitive Memory Engine creates")
    print("   AI systems with human-like memory, understanding, and continuity!")

    print("\n📖 Next steps:")
    print("   1. Complete implementation of core modules")
    print("   2. Run setup.py to configure your environment")
    print("   3. Test with real conversations using your local LLM")

if __name__ == "__main__":
    asyncio.run(main())
