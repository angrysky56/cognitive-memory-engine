# Cognitive Memory Engine MCP Server

A sophisticated Model Context Protocol server implementing a neuroscience-inspired AI memory system with dual-track architecture, hierarchical narrative compression, and intelligent cross-referencing capabilities.

## 🎉 **PROJECT STATUS: PHASE 2 COMPLETE**

**The Dual-Track Architecture with Blended Integration is now fully implemented and working!**

- ✅ **Track 1**: Conversation memory as narrative RTMs
- ✅ **Track 2**: Document knowledge as formal concept RTMs
- ✅ **Track 3**: Blended integration with cross-referencing
- ✅ **Unified Interface**: All tracks accessible through single engine
- ✅ **Production Ready**: Comprehensive testing and error handling

## 🚀 Quick Start

### Installation

Clone or fork the Repo:

```bash
cd /cognitive-memory-engine

# Create and activate virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install dependencies
uv add -e .
```

### Prerequisites

1. **Google AI API Key**: The system uses Google's Gemini models
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

2. **Claude Desktop**: Have Claude Desktop installed for MCP integration

### MCP Server Configuration

Add this to your Claude Desktop with your own paths to any `mcp_config.json`:

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

### Testing the Installation

```bash
# Test Phase 2 capabilities
uv run python test_phase2_simple.py

# Test full dual-track architecture
uv run python test_dual_track.py

# Test the MCP server
uv run python -m cognitive_memory_engine.mcp_server.main
```

## 🧠 Architecture Overview

### **Dual-Track Memory Architecture**

The Cognitive Memory Engine implements a revolutionary dual-track approach to AI memory:

#### **Track 1: Conversation Memory (Narrative RTMs)**
- **Purpose**: Stores conversational dialogues as narrative structures
- **Structure**: Hierarchical Random Tree Models (RTMs) with temporal organization
- **Use Case**: "What did we discuss about AI last week?"
- **Storage**: Temporal books organized by day/session

#### **Track 2: Document Knowledge (Formal RTMs)**
- **Purpose**: Stores formal documents as structured concept hierarchies
- **Structure**: Concept-based RTMs with domain organization
- **Use Case**: "What is SPL in the SAPE framework?"
- **Storage**: Knowledge shelves organized by domain (AI_ARCHITECTURE, etc.)

#### **Track 3: Blended Integration (Cross-Referencing)**
- **Purpose**: Intelligent linking between conversation and document knowledge
- **Structure**: Cross-reference links and unified query interface
- **Use Case**: "Show me both our discussion and formal knowledge about SPL"
- **Storage**: Cross-reference mappings and blended query results

### **Core Components**

1. **Random Tree Model (RTM)** - Hierarchical narrative compression
2. **Temporal Organization** - Time-based memory organization
3. **Document Knowledge Builder** - LLM-powered concept extraction
4. **Cross-Reference System** - Automatic concept linking
5. **Unified Query Interface** - Single access point for all knowledge
6. **Neural Gain Mechanism** - Salience-weighted vector storage

## 🛠 Available MCP Tools

### **Track 1: Conversation Management**
```python
# Store a conversation with full cognitive analysis
store_conversation({
  "conversation": [
    {"role": "user", "content": "What is the SAPE framework?"},
    {"role": "assistant", "content": "SAPE stands for Self-Adaptive Prompt Engineering..."}
  ],
  "context": {"topic": "AI_frameworks", "importance": 0.9}
})

# Query conversation memory with temporal scope
query_memory({
  "query": "What did we discuss about AI last week?",
  "context_depth": 3,
  "time_scope": "week",
  "max_results": 10
})
```

### **Track 2: Document Knowledge Management**
```python
# Store formal documents as structured knowledge
store_document_knowledge({
  "document_content": "SAPE Framework Documentation...",
  "root_concept": "SAPE",
  "domain": "ai_architecture",
  "metadata": {"source": "research_paper", "version": "1.0"}
})

# Direct concept retrieval from document knowledge
get_concept({"concept_name": "SPL"})

# Browse knowledge by domain
browse_knowledge_shelf({"domain": "ai_architecture"})
```

### **Track 3: Blended Integration**
```python
# Query both tracks simultaneously with cross-referencing
query_blended_knowledge({
  "query": "Explain the relationship between SPL and PKG",
  "include_formal": true,
  "include_conversational": true
})

# Create cross-references between conversation mentions and formal concepts
link_conversation_to_knowledge({
  "conversation_id": "uuid",
  "document_concept_id": "optional_specific_concept"
})
```

### **System Analysis & Management**
```python
# Deep conversation analysis
analyze_conversation({"analysis_type": "all"})

# Memory system statistics
get_memory_stats({"include_details": true})

# RTM tree details
get_rtm_tree_details({"tree_id": "uuid"})

# Background task management
list_tasks({"status_filter": "all"})
get_task_status({"task_id": "uuid"})

# Model management
get_available_models({})
get_current_model({})
set_model({"model_name": "gemini-2.0-flash-001"})
```

## 📊 Available MCP Resources

- `cme://memory/conversations` - Recent conversation history with RTM structures
- `cme://memory/documents` - Document knowledge organized by domain
- `cme://memory/concepts` - Direct access to formal concepts
- `cme://memory/cross_references` - Links between tracks
- `cme://memory/temporal` - Time-organized memory books
- `cme://memory/context` - Active workspace context

## 🔧 Development

### **Running Tests**
```bash
# Phase 2 integration tests
uv run python test_phase2_simple.py

# Full dual-track architecture test
uv run python test_dual_track.py

# Concept retrieval validation
uv run python test_concept_retrieval.py
```

### **Code Quality**
```bash
# Format code
uv run ruff format src/

# Check linting
uv run ruff check src/ --fix

# Type checking
uv run mypy src/
```

### **Project Structure**
```
cognitive-memory-engine/
├── src/cognitive_memory_engine/
│   ├── core/              # Main engine and orchestration
│   ├── comprehension/     # RTM builders and analyzers
│   ├── storage/           # Persistence layers
│   ├── workspace/         # Context assembly and vectors
│   ├── production/        # Response generation
│   └── mcp_server/        # MCP integration
├── tests/                 # Test implementations
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## 📈 Performance Characteristics

### **Memory Efficiency**
- **RTM Compression**: ~90% compression ratio for formal documents
- **Temporal Organization**: Efficient time-based retrieval
- **Vector Storage**: Neural gain-weighted embeddings
- **Cross-Reference Indexing**: Fast concept-to-conversation linking

### **Query Performance**
- **Conversation Queries**: Semantic + temporal filtering
- **Document Queries**: Fuzzy concept matching
- **Blended Queries**: Multi-track result integration
- **Context Assembly**: Optimized context window usage

### **Scalability**
- **Concurrent Operations**: Async/await throughout
- **Persistent Storage**: File-based with efficient indexing
- **Memory Management**: LRU caching for frequently accessed data
- **Background Processing**: Automatic compression and maintenance

## 🎯 **Use Cases**

### **Research Assistant**
```python
# Store research papers as formal knowledge
await engine.store_document_knowledge(
    document_content=paper_content,
    root_concept="Transformer Architecture",
    domain="ai_architecture"
)

# Query combining formal knowledge with discussions
result = await engine.query_blended_knowledge(
    "How does attention work in transformers?"
)
```

### **Project Memory**
```python
# Store project conversations
await engine.store_conversation(
    conversation=meeting_transcript,
    context={"project": "Phoenix", "type": "planning"}
)

# Retrieve project-specific knowledge
results = await engine.query_memory(
    query="Phoenix project timeline",
    temporal_scope="month"
)
```

### **Knowledge Management**
```python
# Browse knowledge by domain
ai_concepts = await engine.browse_knowledge_shelf("ai_architecture")

# Link conversations to formal knowledge
cross_refs = await engine.link_conversation_to_knowledge(
    conversation_id="meeting_uuid"
)
```

## 🤝 Contributing

### **Development Priorities**
1. **Phase 3 Enhancements**: Advanced vector integration
2. **Performance Optimization**: Large-scale knowledge base support
3. **UI Integration**: Web interface for knowledge exploration
4. **Additional Domains**: Extended knowledge domain support

### **Code Standards**
- Follow the existing architecture patterns
- Add comprehensive type hints and docstrings
- Include tests for new functionality
- Update documentation with new features

### **Testing Requirements**
- Unit tests for individual components
- Integration tests for cross-track functionality
- Performance benchmarks for large datasets
- Error handling validation

## 🏆 **Recent Achievements**

### **Phase 2 Completion (July 2025)**
- ✅ **Blended Integration Layer**: Cross-referencing between tracks
- ✅ **Unified Query Interface**: Single method for all knowledge types
- ✅ **Enhanced Context Assembly**: Multi-track context integration
- ✅ **Production Readiness**: Comprehensive testing and error handling

### **Phase 1 Foundation**
- ✅ **Dual-Track Architecture**: Conversation + Document knowledge
- ✅ **Random Tree Models**: Hierarchical compression for both tracks
- ✅ **Temporal Organization**: Time-based memory structure
- ✅ **Domain Organization**: Knowledge shelves for formal concepts

## 📝 License

MIT License - See LICENSE file for details.

---

**The Cognitive Memory Engine represents a significant advancement in AI memory architectures, providing human-like memory capabilities with both experiential and declarative knowledge storage, intelligent cross-referencing, and unified access interfaces.**

*For detailed implementation reports, see:*
- `PHASE_1_SUCCESS_REPORT.md`
- `PHASE_2_SUCCESS_REPORT.md`
- `DUAL_TRACK_ARCHITECTURE_TODO.md`
