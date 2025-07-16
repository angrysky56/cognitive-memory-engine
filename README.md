# Cognitive Memory Engine MCP Server

A sophisticated Model Context Protocol server implementing a neuroscience-inspired AI memory system with hierarchical narrative compression, temporal organization, and neural gain mechanisms.

## 🚀 Quick Start

### Installation

```bash
cd /home/ty/Repositories/ai_workspace/cognitive-memory-engine

# Create and activate virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install dependencies
uv add -e .
```

### Prerequisites

1. **Ollama**: Install and start Ollama with a model
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull the default model
   ollama pull qwen3:latest
   ```

cd /cognitive-memory-engine

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync
```

2. **Claude Desktop**: Have Claude Desktop installed for MCP integration

### MCP Server Configuration

Add this to your Claude Desktop `mcp_config.json`:

```json
{
  "mcpServers": {
    "cognitive-memory-engine": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/cognitive-memory-engine",
        "run",
        "python",
        "-m",
        "cognitive_memory_engine.mcp_server.main"
      ],
      "env": {
        "PYTHONPATH": "/home/ty/Repositories/ai_workspace/cognitive-memory-engine/src",
        "CME_DATA_DIR": "/home/ty/cme_data",
        "CME_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Testing the Server

You can test the MCP server directly:

```bash
# Test the server
uv run python -m cognitive_memory_engine.mcp_server.main
```

## 🧠 Architecture Overview

The Cognitive Memory Engine implements several advanced concepts:

### ✅ **Implemented Components**

1. **Random Tree Model (RTM)** - Hierarchical narrative compression
   - Complete algorithm implementation in `comprehension/narrative_tree_builder.py`
   - Uses local LLM (Ollama) for intelligent summarization
   - Builds tree structures with configurable branching and depth

2. **Temporal Organization** - Time-based memory organization
   - Sophisticated temporal books and shelves system
   - Automatic compression and archiving
   - Theme extraction and persistence across time

3. **Type System** - Comprehensive data structures
   - Well-designed dataclasses for all components
   - Proper enum definitions and relationships
   - Full type hints throughout

4. **Core Engine** - Unified orchestration
   - Wires all components together
   - Proper async initialization
   - Real implementations (no mocks)

5. **MCP Server** - Model Context Protocol interface
   - Complete tool and resource definitions
   - Proper error handling and logging
   - Ready for Claude Desktop integration

### 🔄 **Partially Implemented**

1. **Vector Storage** - ChromaDB integration with neural gain
   - Basic ChromaDB wrapper exists
   - Neural gain mechanism designed but needs completion
   - Salience-weighted embeddings need implementation

2. **Context Assembly** - Intelligent context retrieval
   - Framework exists with strategy patterns
   - Hybrid retrieval algorithms need completion
   - Context window optimization needs implementation

3. **Response Generation** - LLM-powered response creation
   - Ollama integration framework exists
   - Prompt templates need completion
   - Social modulation concepts defined but not implemented

### ❌ **Not Yet Implemented**

1. **RTM Graph Store** - NetworkX-based persistence for RTM trees
2. **Complete Vector Manager** - Full salience-weighted vector operations
3. **Advanced Context Assembly** - Complete hybrid retrieval implementation
4. **Social Governance** - Trust and formality modeling

## 🛠 Available MCP Tools

### `store_conversation`
Store a conversation with full cognitive analysis.
```json
{
  "conversation": [
    {"role": "user", "content": "I'm working on the Phoenix project"},
    {"role": "assistant", "content": "What aspects need attention?"}
  ],
  "context": {
    "topic": "project_management",
    "importance": 0.8
  }
}
```

### `query_memory`
Search memory using semantic and temporal constraints.
```json
{
  "query": "What was discussed about the Phoenix project?",
  "context_depth": 3,
  "time_scope": "week",
  "max_results": 10
}
```

### `generate_response`
Generate contextually aware responses using memory.
```json
{
  "prompt": "What's the status of our Q3 timeline?",
  "context_depth": 3,
  "response_style": "analytical"
}
```

### `analyze_conversation`
Perform deep analysis of conversation patterns.
```json
{
  "analysis_type": "all"
}
```

### `get_memory_stats`
Get comprehensive memory system statistics.
```json
{
  "include_details": true
}
```

## 📊 Available MCP Resources

- `cme://memory/conversations` - Recent conversation history
- `cme://memory/narratives` - Hierarchical narrative structures
- `cme://memory/temporal` - Time-organized memory books
- `cme://memory/context` - Active workspace context

## 🔧 Development

### Running Tests
```bash
uv run pytest src/cognitive_memory_engine/tests/
```

### Code Quality
```bash
# Format code
uv run ruff format src/

# Check linting
uv run ruff check src/ --fix

# Type checking
uv run mypy src/
```

### Adding New Features

See `TODO_IMPLEMENTATION.md` for a comprehensive list of what needs to be implemented.

Priority order:
1. Complete RTM Graph Store implementation
2. Finish Vector Manager salience weighting
3. Complete Context Assembler hybrid retrieval
4. Add comprehensive error handling

## 📈 Performance Considerations

The engine uses several optimizations:
- Async/await throughout for concurrent operations
- ChromaDB for efficient vector storage
- File-based indexes for temporal organization
- LRU caching for frequently accessed data
- Background compression for old memories

## 🤝 Contributing

1. Check `TODO_IMPLEMENTATION.md` for current priorities
2. Follow the existing architecture patterns
3. Add comprehensive type hints and docstrings
4. Include tests for new functionality
5. Update this README with new features

## 📝 License

MIT License - See LICENSE file for details.
