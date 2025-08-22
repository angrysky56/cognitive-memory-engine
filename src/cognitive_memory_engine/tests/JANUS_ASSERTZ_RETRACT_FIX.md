# Janus assertz/retract Fix Documentation

## Problem Identified

The code was incorrectly using `janus.assertz()` and `janus.retract()` functions that don't exist in the Janus Python-SWI-Prolog interface.

## Root Cause

The Janus Python interface (`janus_swi`) doesn't provide direct `assertz` and `retract` functions. These are Prolog predicates that must be called through the query interface.

## Incorrect Usage (Before Fix)

```python
# WRONG - These functions don't exist in Janus
janus.assertz(prolog_rule)
janus.retract(prolog_rule)
```

## Correct Usage (After Fix)

```python
# CORRECT - Use query_once to execute Prolog queries
janus.query_once(f"assertz(({prolog_rule}))")
janus.query_once(f"retract(({prolog_rule}))")

# For simple facts (without parentheses in rule):
janus.query_once(f"assertz({fact})")
janus.query_once(f"retract({fact})")
```

## Files Fixed

- `/src/cognitive_memory_engine/semantic/prolog_processor.py`
  - Line 255: Fixed assertz in `_initialize_knowledge_base()`
  - Line 270: Fixed assertz for basic facts
  - Line 352: Fixed assertz in `extract_relations()`
  - Line 401: Fixed assertz in `validate_logic()`
  - Line 409: Fixed retract in `validate_logic()`
  - Line 502: Fixed assertz in `add_domain_rules()`

## Key Points About Janus Interface

1. **No direct assertz/retract functions**: Janus doesn't expose these as Python methods
2. **Use query_once()**: All Prolog operations must go through the query interface
3. **String formatting**: Prolog terms must be properly formatted as strings
4. **Parentheses for rules**: Complex rules should be wrapped in parentheses: `assertz((rule))`
5. **Error handling**: Failed queries will raise `janus.PrologError`

## Example Usage Patterns

### Adding a fact:
```python
janus.query_once("assertz(likes(mary, food))")
```

### Adding a rule:
```python
rule = "happy(X) :- rich(X)"
janus.query_once(f"assertz(({rule}))")
```

### Retracting facts:
```python
janus.query_once("retract(likes(mary, food))")
janus.query_once("retractall(likes(_, _))")  # Remove all likes/2 facts
```

### Querying the knowledge base:
```python
results = list(janus.query("likes(X, Y)"))
for result in results:
    print(f"{result['X']} likes {result['Y']}")
```

## References

- [SWI-Prolog Janus Documentation](https://www.swi-prolog.org/pldoc/package/janus.html)
- [Calling Prolog from Python](https://www.swi-prolog.org/pldoc/man?section=janus-call-prolog)
- [SWI-Prolog assertz/1 Documentation](https://www.swi-prolog.org/pldoc/man?predicate=assertz/1)
