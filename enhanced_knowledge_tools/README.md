# Enhanced Knowledge Tools

This package provides enhanced knowledge ingestion capabilities for the cognitive memory engine.

## Features

- **Direct URL fetching**: Store knowledge by fetching content directly from URLs with intelligent merging
- **Search aggregation**: Store knowledge by searching and aggregating results from multiple sources  
- **Concept enhancement**: Enhance existing concepts with additional information from multiple sources
- **Intelligent merging**: Automatically detect and merge with existing concepts to avoid duplication
- **Source tracking**: Comprehensive metadata tracking for all knowledge sources

## Components

### `enhanced_server_tools.py`
Main implementation of the enhanced knowledge tools. Contains:
- `EnhancedKnowledgeServerTools` class with all enhanced capabilities
- Helper methods for content fetching, aggregation, and merging
- Tool definitions for MCP server integration

### `__init__.py`
Package initialization with clean imports for easy integration.

## Integration

The enhanced tools are automatically integrated into the MCP server through the main.py import system. They provide the following MCP tools:

1. `store_knowledge_from_url` - Store knowledge from a URL with intelligent merging
2. `store_knowledge_from_search` - Store knowledge from search results with aggregation
3. `enhance_existing_concept` - Enhance existing concepts with additional sources

## Architecture

The enhanced tools follow these principles:
- **DRY**: Single implementation in `enhanced_server_tools.py`
- **KISS**: Simple, clear interfaces and implementations
- **Clean imports**: Proper error handling for missing dependencies
- **Modern typing**: Uses Python 3.12+ typing syntax (`dict[str, Any]` instead of `Dict[str, Any]`)

## Backup Files

Previous implementations have been preserved as `.bak` files for reference:
- `enhanced_mcp_tools.py.bak`
- `enhanced_knowledge_ingestion.py.bak`
- `integration.py.bak`
- `integrate_with_mcp_server.py.bak`

These can be safely removed once the new implementation is verified to be working correctly.
