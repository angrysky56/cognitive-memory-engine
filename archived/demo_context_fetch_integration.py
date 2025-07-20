#!/usr/bin/env python3
"""
Demo: Context Assembler with Fetch Integration

This demonstrates how the ContextAssembler can now integrate fresh fetched
content for sub-client AI processing without requiring AI tokens for data
ingestion.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cognitive_memory_engine.workspace.context_assembler import ContextAssembler
from cognitive_memory_engine.mcp_server.enhanced_server_tools import initialize_enhanced_knowledge_tools

async def demo_context_assembler_with_fetch():
    """Demonstrate context assembly with real-time fetch integration."""

    print("🚀 Context Assembler + Fetch Integration Demo")
    print("=" * 50)

    # Mock enhanced knowledge tools for demo
    class MockEnhancedTools:
        async def _perform_search(self, query, max_results):
            return [
                {
                    'content': f'Fresh search result for "{query}" - Latest developments and current information',
                    'title': f'Current: {query}',
                    'url': f'https://example.com/fresh/{query.replace(" ", "-")}',
                    'source': 'web_search'
                }
            ]

        async def _fetch_content_from_url(self, url):
            return f'Fresh content fetched from {url} - This is real-time data that bypasses AI processing and is ready for sub-client consumption.'

    # Mock storage components
    class MockVectorManager:
        async def query_similar_vectors(self, query_text, collection_name, top_k, salience_threshold):
            return [{
                'document': f'Stored knowledge about: {query_text}',
                'salience_score': 0.8,
                'metadata': {
                    'node_id': 'stored_001',
                    'tree_id': 'knowledge_tree_1',
                    'summary': 'Stored knowledge summary'
                }
            }]

    # Initialize context assembler with fetch capabilities
    enhanced_tools = MockEnhancedTools()
    vector_manager = MockVectorManager()

    context_assembler = ContextAssembler(
        vector_manager=vector_manager,
        enhanced_knowledge_tools=enhanced_tools,  # This enables fetch integration
        max_context_length=4096
    )

    print("✅ Context Assembler initialized with fetch capabilities")

    # Demo 1: Standard retrieval (no fetch)
    print("\n📚 Demo 1: Standard Hybrid Retrieval")
    context1 = await context_assembler.assemble_context(
        query="machine learning algorithms",
        strategy="hybrid"
    )
    print(f"Retrieved nodes: {len(context1.retrieved_nodes)}")
    print(f"Sources: {[node.metadata.get('source', 'stored') for node in context1.retrieved_nodes]}")

    # Demo 2: Hybrid with fetch (time-sensitive query)
    print("\n🔄 Demo 2: Hybrid with Fetch (time-sensitive)")
    context2 = await context_assembler.assemble_context(
        query="latest AI developments 2025",
        strategy="hybrid_with_fetch"
    )
    print(f"Retrieved nodes: {len(context2.retrieved_nodes)}")
    print(f"Sources: {[node.metadata.get('source', 'stored') for node in context2.retrieved_nodes]}")
    fresh_nodes = [node for node in context2.retrieved_nodes if node.metadata.get('is_fresh_content')]
    print(f"Fresh content nodes: {len(fresh_nodes)}")

    # Demo 3: Pure fetch strategy
    print("\n⚡ Demo 3: Fetch-Enhanced Retrieval")
    context3 = await context_assembler.assemble_context(
        query="current events breaking news",
        strategy="fetch_enhanced"
    )
    print(f"Retrieved nodes: {len(context3.retrieved_nodes)}")
    fresh_nodes3 = [node for node in context3.retrieved_nodes if node.metadata.get('is_fresh_content')]
    stored_nodes3 = [node for node in context3.retrieved_nodes if not node.metadata.get('is_fresh_content')]
    print(f"Fresh content: {len(fresh_nodes3)}, Stored content: {len(stored_nodes3)}")

    # Demo 4: Show capabilities
    print("\n🔧 Demo 4: Current Capabilities")
    capabilities = await context_assembler.get_current_context()
    print(f"Available strategies: {capabilities['available_strategies']}")
    print(f"Fetch capabilities: {capabilities['fetch_capabilities']}")

    print("\n✨ Key Benefits:")
    print("  🎯 Fresh content fetched directly without AI token overhead")
    print("  🔄 Intelligent mixing of stored and real-time data")
    print("  ⚡ Context optimized for sub-client AI processing")
    print("  📊 Salience-based prioritization of fresh vs stored content")
    print("  🎛️ Multiple retrieval strategies for different use cases")

    print("\n🎉 Context Assembler + Fetch Integration Ready!")

if __name__ == "__main__":
    asyncio.run(demo_context_assembler_with_fetch())
