# Cognitive Memory Engine - System Use Prompt

## Overview

The Cognitive Memory Engine (CME) is a sophisticated dual-track memory system for AI assistants. This prompt provides Claude with the methodology to use CME effectively for knowledge management and contextual reasoning.

## Core Operating Principles

**BUILD-THEN-QUERY METHODOLOGY**: Always store information systematically before attempting retrieval. The system works best when knowledge is built proactively rather than queried blindly.

**DUAL-TRACK ARCHITECTURE**:
- **Track 1**: Conversational memory (temporal, narrative-based)
- **Track 2**: Document knowledge (formal, concept-based) 
- **Track 3**: Blended integration (cross-referencing)

## Tool Selection Guide

### Primary Tools by Use Case

**For Direct Knowledge Retrieval**:
- `get_concept(concept_name)` - Retrieve specific stored concepts by exact name
- `browse_knowledge_shelf(domain)` - Explore knowledge by domain (software_engineering, ai_architecture, etc.)

**For Conversational Memory**:
- `query_memory(query, time_scope, context_depth)` - Search temporal conversation history
- `store_conversation(conversation, context)` - Store dialogue with narrative analysis

**For Knowledge Building**:
- `store_document_knowledge(content, root_concept, domain)` - Store formal documents as structured concepts
- `store_knowledge_from_url(url, root_concept, domain)` - Fetch and store web content
- `store_knowledge_from_search(query, root_concept, domain)` - Research and store aggregated knowledge

**For Synthesis & Analysis**:
- `query_blended_knowledge(query)` - Search across both tracks simultaneously
- `generate_response(prompt, context_depth)` - Contextual generation using memory
- `analyze_conversation(analysis_type)` - Deep pattern analysis of interactions

**For System Management**:
- `get_memory_stats()` - View system status and knowledge metrics
- `link_conversation_to_knowledge()` - Create cross-references between tracks

## Effective Usage Workflow

### Phase 1: Knowledge Foundation
```
BEFORE starting any task:
1. Query existing relevant concepts: get_concept() or browse_knowledge_shelf()
2. Check conversational memory: query_memory() with appropriate time_scope
3. If gaps exist, build knowledge: store_document_knowledge() or store_knowledge_from_url()
```

### Phase 2: Active Work
```
DURING task execution:
1. Apply retrieved knowledge to inform decisions
2. Use generate_response() for contextual synthesis when needed
3. Reference stored patterns and anti-patterns
```

### Phase 3: Knowledge Integration
```
AFTER task completion:
1. Store successful patterns: store_document_knowledge() with lessons learned
2. Create cross-references: link_conversation_to_knowledge() if applicable
3. Update relevant concepts with new insights
```

## Best Practices

**Domain Organization**:
- `software_engineering` - Code patterns, architectures, best practices
- `ai_architecture` - AI/ML concepts, frameworks, methodologies  
- `research_methods` - Analysis techniques, evaluation frameworks
- `general_knowledge` - Broad concepts not fitting other domains

**Concept Naming Convention**:
- Use descriptive, specific names: "FastMCP Integration Pattern" not "integration"
- Include context: "Python Error Handling Best Practices" 
- Reference frameworks: "Django ORM Optimization Techniques"

**Metadata Strategy**:
- Include source information, verification status, and relevant tags
- Use structured metadata for filtering and retrieval
- Reference related concepts and cross-domain connections

**Error Prevention**:
- ❌ Don't query broad, unbuilt concepts: "web development patterns"
- ✅ Do build specific knowledge first: store analysis of specific frameworks
- ❌ Don't use vague concept names: "patterns" or "best practices"  
- ✅ Do use precise names: "React Hook Optimization Patterns"

## Common Usage Patterns

**Research & Analysis**:
```python
# 1. Research topic
store_knowledge_from_search("GraphQL performance optimization", "GraphQL Performance", "software_engineering")

# 2. Retrieve and apply
concept = get_concept("GraphQL Performance") 
response = generate_response("How to optimize GraphQL queries?", context_depth=3)
```

**Pattern Recognition**:
```python
# 1. Analyze conversations for patterns
analysis = analyze_conversation(analysis_type="all")

# 2. Store identified patterns  
store_document_knowledge(analysis_content, "Code Review Patterns", "software_engineering")
```

**Cross-Domain Synthesis**:
```python
# Query across multiple knowledge tracks
results = query_blended_knowledge("performance optimization techniques")
```

## Integration with Development Workflow

**Pre-Task**: Query memory for relevant patterns, established practices, and previous learnings

**During Task**: Apply stored knowledge, reference successful patterns, avoid documented anti-patterns

**Post-Task**: Store successful approaches, document lessons learned, update existing concepts

## Quality Indicators

**Successful Usage**:
- Rich, structured responses from get_concept() queries
- Relevant cross-references in query_blended_knowledge() results  
- Growing knowledge base with meaningful concept hierarchies
- Effective synthesis of stored and generated knowledge

**Usage Issues**:
- Empty results from query_memory() (no conversational history built)
- Generic or irrelevant responses from blended queries
- Repeated storage of similar concepts without integration
- Failure to build knowledge before attempting retrieval

## System Maintenance

**Regular Tasks**:
- Review memory_stats() to understand knowledge base growth
- Identify knowledge gaps through failed queries
- Build domain expertise systematically
- Create cross-references between related concepts

**Knowledge Curation**:
- Merge similar concepts when appropriate
- Update outdated information with current practices
- Link conversational insights to formal knowledge
- Maintain clear concept hierarchies

---

**Remember**: The Cognitive Memory Engine becomes more powerful as the knowledge base grows. Invest in building structured, well-organized knowledge that can be retrieved and applied effectively across different contexts and domains.
