# Phase 3B-2: LQEPR Unified Query Interface Implementation

## Executive Summary
Implement LogicalQueryProcessor to unify Prolog formal logic + DuckDB graph queries + ChromaDB vector retrieval into elegant LQEPR (Logical Query Enhanced Pattern Retrieval) interface.

**Key Finding**: LMQL is NOT needed. Current architecture is superior.

## Current Architecture Assessment ✅

### Already Available (Ready for Integration)
- **SWI-Prolog**: Working integration in `prolog_processor.py`
- **DuckDB**: Semantic graph store from Phase 3B-1  
- **ChromaDB**: Vector store with neural gain
- **Abstract Interfaces**: `AbstractQueryEngine` defined
- **Type System**: Complete semantic types

### Why LMQL is Unnecessary
Your current approach is more elegant:
- **Prolog**: Handles formal logic reasoning directly
- **DuckDB**: High-performance graph traversal 
- **Python**: Async orchestration with proper error handling
- **LMQL**: Would add unnecessary complexity and dependencies

## Phase 3B-2 Implementation Plan

### Target: LogicalQueryProcessor Class
```python
class LogicalQueryProcessor(AbstractQueryEngine):
    """
    LQEPR Unified Query Interface
    
    Integrates three query modes:
    1. Formal Logic (Prolog) - compositional reasoning
    2. Graph Traversal (DuckDB) - relationship queries  
    3. Vector Similarity (ChromaDB) - neural gain retrieval
    """
```

### Integration Architecture
```
User Query → LogicalQueryProcessor → [Prolog + DuckDB + ChromaDB] → Unified Results
```

## Implementation Steps

### Step 1: Extend Existing PrologProcessor
**File**: `src/cognitive_memory_engine/semantic/prolog_processor.py`
- Add async query methods
- Integrate with DuckDB graph store
- Add result ranking by logical confidence

### Step 2: Create Unified Query Interface  
**File**: `src/cognitive_memory_engine/semantic/logical_query_processor.py`
- Implement `AbstractQueryEngine` interface
- Orchestrate Prolog + DuckDB + ChromaDB
- Handle query routing and result fusion

### Step 3: Update Context Assembler
**File**: `src/cognitive_memory_engine/workspace/context_assembler.py`  
- Replace vector-only queries with LogicalQueryProcessor
- Maintain backward compatibility
- Add logical relevance scoring

### Step 4: Enhanced CLI Interface
**File**: `src/cognitive_memory_engine/cli/query_commands.py`
- Add `cme query-logic "<formal query>"` command
- Interactive query builder with typer + rich
- Visualization of logical reasoning chains

## Prolog-SWI Integration Pattern

### Current Status ✅
Your `prolog_processor.py` already handles SWI-Prolog correctly:
- Graceful fallback for development without SWI-Prolog
- Proper error handling and async patterns
- Ready for extension

### Enhancement Strategy
```python
# Enhance existing PrologProcessor for Phase 3B-2
class EnhancedPrologProcessor(PrologProcessor):
    async def query_with_graph_context(self, 
                                     prolog_query: str,
                                     graph_context: List[str]) -> LogicalResult:
        """Combine Prolog reasoning with DuckDB graph data"""
        # 1. Load graph context into Prolog working memory
        # 2. Execute formal logic query
        # 3. Return structured logical results
```

## Success Metrics

### Functional Requirements
- [ ] Unified query interface works with all three backends
- [ ] Logical queries outperform vector similarity for complex reasoning
- [ ] Backward compatibility maintained
- [ ] Performance ≤ 500ms for complex queries

### Elegance Requirements  
- [ ] Single unified interface replaces multiple query methods
- [ ] Clear separation of concerns (logic vs graph vs vector)
- [ ] Graceful degradation when components unavailable
- [ ] Rich CLI for query building and visualization

## Risk Mitigation

### Low Risk Implementation
- **Extend existing code** (don't replace)
- **Maintain existing interfaces** (backward compatibility)
- **Add new capabilities** (LogicalQueryProcessor as addition)
- **Test incrementally** (verify each integration step)

### Rollback Strategy
- LogicalQueryProcessor is additive
- Existing vector queries remain functional
- Can disable logical processing if issues arise

## Timeline: 1 Week Implementation

### Day 1-2: Enhance PrologProcessor
- Add async graph integration methods
- Test Prolog + DuckDB data flow

### Day 3-4: Implement LogicalQueryProcessor
- Create unified interface
- Implement query routing logic

### Day 5-6: Update Context Assembler
- Integrate logical queries
- Test performance vs existing system

### Day 7: CLI and Documentation
- Add query commands with typer + rich
- Update documentation and examples

## Next Steps

1. **Examine current PrologProcessor implementation**
2. **Design LogicalQueryProcessor class structure**  
3. **Implement unified query routing**
4. **Test with real queries and measure performance**

**Recommendation**: Proceed with LogicalQueryProcessor implementation using existing SWI-Prolog integration. No LMQL needed.
