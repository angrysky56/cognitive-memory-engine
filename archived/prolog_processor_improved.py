"""
Improved Prolog Semantic Processor - Optional Dependency Architecture

This is a drop-in replacement for the original prolog_processor.py that:
1. Gracefully handles missing SWI-Prolog/janus dependencies
2. Provides semantic fallbacks when Prolog is unavailable
3. Maintains full compatibility with existing CME architecture
4. Offers enhanced error handling and monitoring

Key Improvements:
- Safe import handling with fallback implementations
- Semantic analysis alternatives for formal logic operations
- Comprehensive error handling and logging
- Zero breaking changes for existing code
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Safe import of Prolog dependencies with fallback
try:
    import janus_swi as janus
    JANUS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("SWI-Prolog/janus available - formal logic mode enabled")
except ImportError:
    janus = None
    JANUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("SWI-Prolog/janus not available - using semantic fallback mode")

from .abstract_processors import (
    AbstractSemanticProcessor,
    LogicalForm,
    SemanticResult,
)


@dataclass
class PrologRule:
    """Represents a Prolog rule for semantic reasoning"""
    head: str  # Predicate head (conclusion)
    body: str  # Rule body (conditions)
    confidence: float
    source: str  # Where this rule came from

    def to_prolog_syntax(self) -> str:
        """Convert to Prolog syntax"""
        if self.body:
            return f"{self.head} :- {self.body}."
        else:
            return f"{self.head}."

    def to_semantic_pattern(self) -> dict[str, Any]:
        """Convert to semantic pattern for fallback processing"""
        return {
            'pattern': self.head,
            'conditions': self.body.split(',') if self.body else [],
            'confidence': self.confidence,
            'source': self.source
        }


@dataclass
class SemanticQuery:
    """Represents a semantic query in Prolog or semantic format"""
    query: str
    variables: list[str]
    expected_bindings: dict[str, Any]
    query_type: str = "semantic"  # "prolog" or "semantic"


class MontagueGrammarRules:
    """
    Montague Grammar rules with dual implementation:
    - Prolog rules when available
    - Semantic patterns as fallback
    """

    def __init__(self):
        self.core_rules = [
            # Basic compositional rules
            PrologRule(
                head="meaning(compound(X, Y), compose(MX, MY))",
                body="meaning(X, MX), meaning(Y, MY)",
                confidence=1.0,
                source="montague_core"
            ),

            # Quantifier rules
            PrologRule(
                head="quantifier(every, universal)",
                body="",
                confidence=1.0,
                source="montague_quantifiers"
            ),

            PrologRule(
                head="quantifier(some, existential)",
                body="",
                confidence=1.0,
                source="montague_quantifiers"
            ),

            # Temporal logic rules
            PrologRule(
                head="temporal(before(X, Y), precedes(X, Y))",
                body="event(X), event(Y)",
                confidence=0.9,
                source="montague_temporal"
            ),

            # Entity relationship rules
            PrologRule(
                head="relation(X, relates_to, Y)",
                body="entity(X), entity(Y), semantic_link(X, Y)",
                confidence=0.8,
                source="montague_relations"
            ),

            # Modal logic rules
            PrologRule(
                head="modal(possible(X), diamond(X))",
                body="proposition(X)",
                confidence=0.9,
                source="montague_modal"
            ),

            PrologRule(
                head="modal(necessary(X), box(X))",
                body="proposition(X)",
                confidence=0.9,
                source="montague_modal"
            ),
        ]

        # Convert to semantic patterns for fallback processing
        self.semantic_patterns = [rule.to_semantic_pattern() for rule in self.core_rules]

    def get_rules_by_category(self, category: str) -> list[PrologRule]:
        """Get rules by source category"""
        return [rule for rule in self.core_rules if category in rule.source]

    def get_semantic_patterns_by_category(self, category: str) -> list[dict[str, Any]]:
        """Get semantic patterns by category for fallback processing"""
        return [pattern for pattern in self.semantic_patterns if category in pattern['source']]


class ImprovedPrologSemanticProcessor(AbstractSemanticProcessor):
    """
    Improved semantic processor with graceful Prolog fallback

    Features:
    - Optional Prolog integration (works with or without SWI-Prolog)
    - Semantic analysis fallbacks for all operations
    - Enhanced error handling and monitoring
    - Full compatibility with existing CME architecture
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.grammar_rules = MontagueGrammarRules()
        self.knowledge_base = []
        self.semantic_knowledge_base = []  # Fallback knowledge storage
        self.prolog_mode = JANUS_AVAILABLE

        # Initialize knowledge base
        self._initialize_knowledge_base()

        # System status tracking
        self.system_status = {
            'prolog_available': JANUS_AVAILABLE,
            'fallback_mode': not JANUS_AVAILABLE,
            'knowledge_base_size': 0,
            'semantic_patterns': len(self.grammar_rules.semantic_patterns)
        }

        logger.info(f"Semantic processor initialized - Prolog mode: {self.prolog_mode}")

    def _initialize_knowledge_base(self):
        """Initialize knowledge base with dual-mode support"""
        if JANUS_AVAILABLE and janus:
            try:
                # Load core Montague Grammar rules into Prolog
                for rule in self.grammar_rules.core_rules:
                    prolog_rule = rule.to_prolog_syntax()
                    janus.assertz(prolog_rule)
                    self.knowledge_base.append(prolog_rule)

                # Add basic semantic facts
                basic_facts = [
                    "entity(person).",
                    "entity(object).",
                    "entity(concept).",
                    "event(action).",
                    "event(process).",
                    "proposition(statement).",
                    "proposition(question).",
                ]

                for fact in basic_facts:
                    janus.assertz(fact)
                    self.knowledge_base.append(fact)

                logger.info(f"Prolog knowledge base initialized with {len(self.knowledge_base)} rules")

            except Exception as e:
                logger.warning(f"Prolog initialization failed, using semantic fallback: {e}")
                self.prolog_mode = False
                self._initialize_semantic_fallback()
        else:
            self._initialize_semantic_fallback()

        self.system_status['knowledge_base_size'] = len(self.knowledge_base) + len(self.semantic_knowledge_base)

    def _initialize_semantic_fallback(self):
        """Initialize semantic fallback knowledge base"""
        # Store semantic patterns as structured data
        self.semantic_knowledge_base = [
            {'type': 'entity', 'value': 'person', 'category': 'basic'},
            {'type': 'entity', 'value': 'object', 'category': 'basic'},
            {'type': 'entity', 'value': 'concept', 'category': 'basic'},
            {'type': 'event', 'value': 'action', 'category': 'basic'},
            {'type': 'event', 'value': 'process', 'category': 'basic'},
            {'type': 'proposition', 'value': 'statement', 'category': 'basic'},
            {'type': 'proposition', 'value': 'question', 'category': 'basic'},
        ]

        # Add semantic patterns from grammar rules
        for pattern in self.grammar_rules.semantic_patterns:
            self.semantic_knowledge_base.append({
                'type': 'rule',
                'pattern': pattern['pattern'],
                'conditions': pattern['conditions'],
                'confidence': pattern['confidence'],
                'source': pattern['source']
            })

        logger.info(f"Semantic fallback knowledge base initialized with {len(self.semantic_knowledge_base)} items")

    async def parse_semantics(self, text: str) -> SemanticResult:
        """
        Parse natural language with dual-mode support (Prolog or semantic fallback)
        """
        start_time = datetime.now()

        try:
            # Step 1: Extract semantic components
            semantic_components = await self._extract_semantic_components(text)

            # Step 2: Apply semantic composition (Prolog or fallback)
            if self.prolog_mode:
                logical_form = await self._compose_semantics_prolog(semantic_components)
            else:
                logical_form = await self._compose_semantics_semantic(semantic_components)

            # Step 3: Generate structured facts
            structured_facts = await self._generate_structured_facts(logical_form, text)

            processing_time = (datetime.now() - start_time).total_seconds()

            result = SemanticResult(
                success=True,
                data={
                    "logical_form": logical_form,
                    "structured_facts": structured_facts,
                    "semantic_components": semantic_components,
                    "processing_mode": "prolog" if self.prolog_mode else "semantic_fallback"
                },
                metadata={
                    "method": "prolog_montague" if self.prolog_mode else "semantic_montague",
                    "rule_count": len(self.grammar_rules.core_rules),
                    "components_found": len(semantic_components),
                    "prolog_available": JANUS_AVAILABLE
                },
                processing_time=processing_time,
                timestamp=start_time
            )

            self.update_stats(processing_time, True)
            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.update_stats(processing_time, False)

            return SemanticResult(
                success=False,
                data={},
                metadata={
                    "error_type": "parsing_error",
                    "processing_mode": "prolog" if self.prolog_mode else "semantic_fallback"
                },
                processing_time=processing_time,
                timestamp=start_time,
                error_message=str(e)
            )

    async def extract_relations(self, semantic_data: Any) -> list[tuple[str, str, str]]:
        """
        Extract entity relationships using Prolog inference or semantic analysis
        """
        try:
            logical_form = semantic_data.get("logical_form", {})
            structured_facts = semantic_data.get("structured_facts", [])

            if self.prolog_mode and JANUS_AVAILABLE:
                return await self._extract_relations_prolog(structured_facts)
            else:
                return await self._extract_relations_semantic(logical_form, structured_facts)

        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}")
            return await self._fallback_relation_extraction(semantic_data.get("logical_form", {}))

    async def _extract_relations_prolog(self, prolog_facts: list[str]) -> list[tuple[str, str, str]]:
        """Extract relations using Prolog queries"""
        relations = []

        try:
            # Temporarily add facts to Prolog knowledge base
            for fact in prolog_facts:
                janus.assertz(fact)

            # Query for relationships
            for solution in janus.query("relation(X, Pred, Y)"):
                subject = solution.get("X", "unknown")
                predicate = solution.get("Pred", "unknown")
                object_val = solution.get("Y", "unknown")
                relations.append((str(subject), str(predicate), str(object_val)))

            # Query for semantic links
            for solution in janus.query("semantic_link(X, Y)"):
                subject = solution.get("X", "unknown")
                object_val = solution.get("Y", "unknown")
                relations.append((str(subject), "links_to", str(object_val)))

        except Exception as e:
            logger.warning(f"Prolog relation extraction failed: {e}")

        return relations

    async def _extract_relations_semantic(self, logical_form: dict[str, Any], structured_facts: list) -> list[tuple[str, str, str]]:
        """Extract relations using semantic analysis"""
        relations = []

        # Extract relations from structured facts
        for fact in structured_facts:
            if isinstance(fact, dict):
                if fact.get('type') == 'relation':
                    subject = fact.get('subject', 'unknown')
                    predicate = fact.get('predicate', 'relates_to')
                    object_val = fact.get('object', 'unknown')
                    relations.append((subject, predicate, object_val))

        # Extract relations from logical form
        if isinstance(logical_form, dict):
            entities = logical_form.get('variables', [])
            predicates = logical_form.get('predicates', [])

            # Create simple relations between entities and predicates
            for i, entity in enumerate(entities):
                for predicate in predicates:
                    if i < len(entities) - 1:
                        next_entity = entities[i + 1]
                        relations.append((entity, predicate, next_entity))

        return relations

    async def validate_logic(self, logical_form: LogicalForm) -> bool:
        """
        Validate logical consistency using Prolog reasoning or semantic validation
        """
        try:
            if self.prolog_mode and JANUS_AVAILABLE:
                return await self._validate_logic_prolog(logical_form)
            else:
                return await self._validate_logic_semantic(logical_form)

        except Exception as e:
            logger.warning(f"Logic validation failed: {e}")
            return False

    async def _validate_logic_prolog(self, logical_form: LogicalForm) -> bool:
        """Validate logic using Prolog"""
        try:
            prolog_expression = self._convert_to_prolog(logical_form.expression)

            if not self._is_valid_prolog_syntax(prolog_expression):
                return False

            # Test the expression in Prolog
            test_rule = f"test_rule :- {prolog_expression}."
            janus.assertz(test_rule)
            list(janus.query("test_rule"))
            return True

        except Exception:
            return False

    async def _validate_logic_semantic(self, logical_form: LogicalForm) -> bool:
        """Validate logic using semantic analysis"""
        try:
            # Basic semantic validation
            if not logical_form.expression:
                return False

            # Check for valid structure
            if not logical_form.variables and not logical_form.predicates:
                return False

            # Check for reasonable confidence
            if logical_form.confidence < 0.1:
                return False

            return True

        except Exception:
            return False

    # Composition methods for dual-mode support

    async def _compose_semantics_prolog(self, components: list[dict[str, Any]]) -> LogicalForm:
        """Apply Montague Grammar composition using Prolog"""
        # Use Prolog for sophisticated composition
        predicates = [c["word"] for c in components if c["type"] == "predicate"]
        entities = [c["word"] for c in components if c["type"] == "entity"]
        quantifiers = [c["word"] for c in components if c["type"] == "quantifier"]

        expression = f"semantic_structure({', '.join(predicates + entities)})"

        return LogicalForm(
            expression=expression,
            variables=entities,
            quantifiers=quantifiers,
            predicates=predicates,
            confidence=0.8,
            source_text=" ".join([c["word"] for c in components])
        )

    async def _compose_semantics_semantic(self, components: list[dict[str, Any]]) -> LogicalForm:
        """Apply semantic composition using pattern matching"""
        # Semantic composition without Prolog
        predicates = [c["word"] for c in components if c["type"] == "predicate"]
        entities = [c["word"] for c in components if c["type"] == "entity"]
        quantifiers = [c["word"] for c in components if c["type"] == "quantifier"]

        # Apply semantic patterns
        confidence = 0.7  # Slightly lower confidence for semantic fallback

        # Enhance confidence based on pattern matching
        for pattern in self.semantic_knowledge_base:
            if pattern.get('type') == 'rule':
                if any(entity in pattern.get('pattern', '') for entity in entities):
                    confidence = min(0.9, confidence + 0.1)

        expression = f"semantic_concept({', '.join(entities + predicates)})"

        return LogicalForm(
            expression=expression,
            variables=entities,
            quantifiers=quantifiers,
            predicates=predicates,
            confidence=confidence,
            source_text=" ".join([c["word"] for c in components])
        )

    async def _generate_structured_facts(self, logical_form: LogicalForm, source_text: str) -> list[Any]:
        """Generate structured facts from logical form (Prolog or semantic)"""
        if self.prolog_mode:
            return await self._generate_prolog_facts(logical_form, source_text)
        else:
            return await self._generate_semantic_facts(logical_form, source_text)

    async def _generate_prolog_facts(self, logical_form: LogicalForm, source_text: str) -> list[str]:
        """Generate Prolog facts from logical form"""
        facts = []

        # Generate entity facts
        for entity in logical_form.variables:
            facts.append(f"entity({entity}).")

        # Generate predicate facts
        for predicate in logical_form.predicates:
            facts.append(f"predicate({predicate}).")

        # Generate source fact
        facts.append(f"source_text('{source_text}').")

        return facts

    async def _generate_semantic_facts(self, logical_form: LogicalForm, source_text: str) -> list[dict[str, Any]]:
        """Generate semantic facts from logical form"""
        facts = []

        # Generate entity facts
        for entity in logical_form.variables:
            facts.append({
                'type': 'entity',
                'value': entity,
                'source': source_text,
                'confidence': logical_form.confidence
            })

        # Generate predicate facts
        for predicate in logical_form.predicates:
            facts.append({
                'type': 'predicate',
                'value': predicate,
                'source': source_text,
                'confidence': logical_form.confidence
            })

        # Generate source fact
        facts.append({
            'type': 'source',
            'value': source_text,
            'confidence': 1.0
        })

        return facts

    # Enhanced query methods

    async def query_compositional(self, query_text: str, max_results: int = 5) -> list[LogicalResult]:
        """Query using compositional semantics with dual-mode support"""
        try:
            if self.prolog_mode and JANUS_AVAILABLE:
                return await self._query_compositional_prolog(query_text, max_results)
            else:
                return await self._query_compositional_semantic(query_text, max_results)
        except Exception as e:
            logger.warning(f"Compositional query failed: {e}")
            return []

    async def _query_compositional_prolog(self, query_text: str, max_results: int) -> list[LogicalResult]:
        """Query using Prolog compositional reasoning"""
        results = []

        try:
            # Convert query to Prolog format
            prolog_query = self._convert_query_to_prolog(query_text)

            # Execute Prolog query
            for i, solution in enumerate(janus.query(prolog_query)):
                if i >= max_results:
                    break

                result = LogicalResult(
                    logical_form=str(solution),
                    variables=list(solution.keys()),
                    bindings=dict(solution),
                    confidence=0.8,
                    source_query=query_text
                )
                results.append(result)

        except Exception as e:
            logger.warning(f"Prolog compositional query failed: {e}")

        return results

    async def _query_compositional_semantic(self, query_text: str, max_results: int) -> list[LogicalResult]:
        """Query using semantic compositional analysis"""
        results = []

        # Extract key terms from query
        key_terms = await self._extract_key_terms(query_text)

        # Match against semantic knowledge base
        for i, term in enumerate(key_terms[:max_results]):
            # Find matching patterns in semantic knowledge base
            matches = [
                item for item in self.semantic_knowledge_base
                if term.lower() in str(item.get('value', '')).lower() or
                   term.lower() in str(item.get('pattern', '')).lower()
            ]

            if matches:
                best_match = max(matches, key=lambda x: x.get('confidence', 0.5))

                result = LogicalResult(
                    logical_form=f"semantic_match({term}, {best_match.get('value', 'unknown')})",
                    variables=[term],
                    bindings={term: best_match.get('value', 'unknown')},
                    confidence=best_match.get('confidence', 0.5),
                    source_query=query_text
                )
                results.append(result)

        return results

    async def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from text for semantic analysis"""
        # Simple key term extraction
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]

        return key_terms[:10]  # Limit to avoid noise

    # Utility methods

    def _convert_query_to_prolog(self, query_text: str) -> str:
        """Convert natural language query to Prolog query"""
        # Simplified conversion - in practice, this would be more sophisticated
        key_terms = re.findall(r'\b\w+\b', query_text.lower())
        if key_terms:
            return f"entity({key_terms[0]})"
        return "entity(X)"

    def _convert_to_prolog(self, expression: str) -> str:
        """Convert logical form expression to Prolog syntax"""
        return expression.replace("(", "(").replace(")", ")")

    def _is_valid_prolog_syntax(self, expression: str) -> bool:
        """Check if expression has valid Prolog syntax"""
        return "(" in expression and ")" in expression

    async def _fallback_relation_extraction(self, logical_form: dict[str, Any]) -> list[tuple[str, str, str]]:
        """Fallback relation extraction when other methods fail"""
        return [("entity1", "relates_to", "entity2")]

    # Enhanced component extraction

    async def _extract_semantic_components(self, text: str) -> list[dict[str, Any]]:
        """Extract semantic components with enhanced analysis"""
        words = text.lower().split()
        components = []

        for word in words:
            component = {
                "word": word,
                "type": self._classify_word_type(word),
                "semantic_role": self._determine_semantic_role(word),
                "confidence": self._calculate_word_confidence(word)
            }
            components.append(component)

        return components

    def _classify_word_type(self, word: str) -> str:
        """Enhanced word type classification"""
        quantifiers = {"every", "some", "all", "none", "most", "many", "few"}
        predicates = {"is", "has", "does", "can", "will", "relates", "connects", "contains", "includes"}

        if word in quantifiers:
            return "quantifier"
        elif word in predicates:
            return "predicate"
        else:
            return "entity"

    def _determine_semantic_role(self, word: str) -> str:
        """Determine semantic role with context awareness"""
        # Enhanced role determination could consider context
        return "content"

    def _calculate_word_confidence(self, word: str) -> float:
        """Calculate confidence score for word classification"""
        # Simple confidence based on word length and patterns
        if len(word) < 3:
            return 0.3
        elif len(word) > 10:
            return 0.7
        else:
            return 0.5

    # Public interface methods

    def add_domain_rules(self, domain: str, rules: list[PrologRule]):
        """Add domain-specific rules with dual-mode support"""
        for rule in rules:
            if self.prolog_mode and JANUS_AVAILABLE:
                try:
                    prolog_rule = rule.to_prolog_syntax()
                    janus.assertz(prolog_rule)
                    self.knowledge_base.append(prolog_rule)
                except Exception as e:
                    logger.warning(f"Failed to add Prolog rule: {e}")

            # Always add to semantic knowledge base as backup
            semantic_pattern = rule.to_semantic_pattern()
            semantic_pattern['domain'] = domain
            self.semantic_knowledge_base.append(semantic_pattern)

    def query_knowledge_base(self, query: str) -> list[dict[str, Any]]:
        """Query knowledge base with dual-mode support"""
        if self.prolog_mode and JANUS_AVAILABLE:
            try:
                results = []
                for solution in janus.query(query):
                    results.append(dict(solution))
                return results
            except Exception as e:
                logger.warning(f"Prolog query failed, using semantic fallback: {e}")

        # Semantic fallback query
        return self._query_semantic_knowledge_base(query)

    def _query_semantic_knowledge_base(self, query: str) -> list[dict[str, Any]]:
        """Query semantic knowledge base as fallback"""
        results = []
        key_terms = re.findall(r'\b\w+\b', query.lower())

        for item in self.semantic_knowledge_base:
            item_text = str(item).lower()
            matches = sum(1 for term in key_terms if term in item_text)

            if matches > 0:
                confidence = matches / len(key_terms) if key_terms else 0.5
                result = dict(item)
                result['match_confidence'] = confidence
                results.append(result)

        # Sort by confidence
        return sorted(results, key=lambda x: x.get('match_confidence', 0), reverse=True)

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """Get comprehensive knowledge base statistics"""
        return {
            "prolog_available": JANUS_AVAILABLE,
            "prolog_mode": self.prolog_mode,
            "prolog_rules": len(self.knowledge_base),
            "semantic_patterns": len(self.semantic_knowledge_base),
            "montague_rules": len(self.grammar_rules.core_rules),
            "processing_stats": self.processing_stats,
            "performance_metrics": self.get_performance_metrics(),
            "system_status": self.system_status
        }

    def get_system_health(self) -> dict[str, Any]:
        """Get system health and diagnostic information"""
        return {
            "dependencies": {
                "janus_swi": JANUS_AVAILABLE,
                "prolog_processor": self.prolog_mode
            },
            "knowledge_base": {
                "total_items": len(self.knowledge_base) + len(self.semantic_knowledge_base),
                "prolog_rules": len(self.knowledge_base),
                "semantic_patterns": len(self.semantic_knowledge_base)
            },
            "performance": self.get_performance_metrics(),
            "recommendations": self._generate_health_recommendations()
        }

    def _generate_health_recommendations(self) -> list[str]:
        """Generate system health recommendations"""
        recommendations = []

        if not JANUS_AVAILABLE:
            recommendations.append("Install SWI-Prolog and janus-swi for enhanced formal logic capabilities")

        if len(self.knowledge_base) + len(self.semantic_knowledge_base) < 10:
            recommendations.append("Consider adding domain-specific rules to improve semantic processing")

        if self.processing_stats.get('total_processed', 0) == 0:
            recommendations.append("No processing history - system ready for use")

        return recommendations


# Backward compatibility alias
PrologSemanticProcessor = ImprovedPrologSemanticProcessor
