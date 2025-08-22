# Cognitive Memory Engine - Simplified Usage Guide

## Core Principle: PROACTIVE PATTERN APPLICATION
The CME learns from successful interactions and proactively applies proven patterns. Focus on what WORKS, not just what was discussed.

## Essential Tools (5 Core Functions)

### 1. **Quick Knowledge Retrieval**
```
get_concept("FastMCP Integration") - Get specific stored knowledge
browse_knowledge_shelf("software_engineering") - Explore domain knowledge
query_blended_knowledge("debugging connection issues") - Search everything
generate_response (chat with the librarian)

```

### 2. **Knowledge Building**
```
store_document_knowledge(content, "React Optimization", "software_engineering")
store_knowledge_from_url(url, "GraphQL Best Practices", "software_engineering")
```

### 3. **Conversation Memory**
```
query_memory("What debugging steps worked for (the subject)?", time_scope="week")
store_conversation(messages, context={"topic": "debugging", "success": true})
```

### 4. **System Status**
```
get_memory_stats() - See what knowledge is available
```

## Automatic Behaviors (No Action Needed)

**Pattern Recognition**: CME automatically identifies when users ask about:
- Debugging/troubleshooting → Suggests proven diagnostic sequences
- Implementation tasks → Retrieves successful approaches
- Technical questions → Pulls relevant stored knowledge

**Success Learning**: When interactions resolve problems successfully, patterns are automatically stored for future use.

**Proactive Suggestions**: CME checks for relevant stored solutions when similar issues arise.

## Usage Workflow

**For Technical Questions:**
1. CME automatically checks for relevant patterns/knowledge
2. If found: Applies proven solutions immediately
3. If not found: Builds knowledge for future use

**For Complex Problems:**
1. Check: `query_blended_knowledge("your problem")`
2. Store solutions: `store_document_knowledge()` with successful approaches
3. Future similar problems will auto-suggest these solutions

**For Learning/Research:**
1. `store_knowledge_from_url()` or `store_document_knowledge()`
2. `get_concept()` for quick retrieval
3. `browse_knowledge_shelf()` for exploration

## Domain Organization
- `software_engineering` - Code, debugging, frameworks, tools
- `ai_architecture` - LLM, ML, AI system design
- `general_knowledge` - Everything else

## Quality Indicators

**Good Usage:**
- Rich responses from `get_concept()`
- Relevant auto-suggestions for technical problems
- Growing knowledge base with practical solutions

**Issues:**
- Empty results (need to build knowledge first)
- Repeated similar questions (patterns not being stored/applied)

## Remember
- **Build knowledge for reuse** - Store successful solutions
- **Name concepts specifically** - "FastMCP Connection Fix" not "connection issue"
- **Focus on what works** - Practical patterns over theoretical knowledge
- **Let CME help proactively** - It will suggest relevant stored solutions

---

**The CME becomes more valuable as it learns patterns of what actually works in practice.**
