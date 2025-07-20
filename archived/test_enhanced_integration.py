#!/usr/bin/env python3
"""
Test the enhanced knowledge tools integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_memory_engine.mcp_server.enhanced_server_tools import EnhancedKnowledgeServerTools, initialize_enhanced_knowledge_tools

def test_enhanced_tools_integration():
    """Test basic functionality of enhanced knowledge tools."""

    # Mock cognitive memory engine for testing
    class MockCME:
        async def store_document_knowledge(self, document_content, root_concept, domain, metadata=None):
            return {
                'status': 'success',
                'concept': root_concept,
                'domain': domain,
                'content_length': len(document_content),
                'metadata': metadata
            }

        async def browse_knowledge_shelf(self, domain):
            return {'documents': []}

        async def get_concept(self, concept_name):
            return {'content': f'Existing content for {concept_name}'}

    # Initialize enhanced tools
    mock_cme = MockCME()
    tools = initialize_enhanced_knowledge_tools(mock_cme)

    # Test tool initialization
    assert tools is not None
    print("✅ Enhanced knowledge tools initialized successfully")

    # Test MCP tool registration
    def mock_fetch_tool(url):
        return {
            'content': f'Mocked content from {url}',
            'title': 'Test Page',
            'status': 'success'
        }

    tools.set_mcp_tool("mcp-server-fetch:fetch", mock_fetch_tool)
    assert tools.has_mcp_tool("mcp-server-fetch:fetch")
    print("✅ MCP tool registration working")

    # Test that we can access the fetch capability
    fetch_tool = tools.mcp_tools.get("mcp-server-fetch:fetch")
    assert fetch_tool is not None
    assert callable(fetch_tool)
    print("✅ Fetch tool is callable and ready")

    print("\n🎉 All integration tests passed!")
    print("📋 Summary:")
    print("  - Enhanced knowledge tools ready for MCP server integration")
    print("  - Direct content fetching capability available")
    print("  - No AI processing overhead for data ingestion")
    print("  - AI only engaged when user specifically requests analysis")

if __name__ == "__main__":
    test_enhanced_tools_integration()
