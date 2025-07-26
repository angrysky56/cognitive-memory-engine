"""
Prolog + Montague Semantic Processor - Phase 3A Superior Architecture

Implements formal logic reasoning using Prolog for Montague Grammar
compositional semantics. Far superior to statistical approaches.

Key Advantages of Prolog + Montague:
- Formal logic representation (not statistical approximation)
- Pattern matching and unification (perfect for semantic parsing)
- Rule-based inference (compositional semantics)
- Declarative programming (elegant and maintainable)

Following AETHELRED principles:
- Prioritize Clarity: Formal logic rules are explicit and readable
- Embrace Simplicity: Prolog's declarative nature eliminates complexity
- Think Structurally: Rules compose naturally into logical hierarchies
- Be Collaborative: Extensible rule base for semantic enhancement
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import LogicalResult, QueryMode
from .abstract_processors import (
    AbstractSemanticProcessor,
    LogicalForm,
    SemanticProcessingError,
    SemanticResult,
)

logger = logging.getLogger(__name__)

# Try to import janus_swi, but handle the case where it's not available
try:
    import janus_swi as janus
    JANUS_AVAILABLE = True
except ImportError:
    janus = None
    JANUS_AVAILABLE = False
    logger.warning("janus_swi not available - Prolog functionality will be disabled")


def check_prolog_installation() -> tuple[bool, str | None]:
    """
    Check if SWI-Prolog is installed and available on the system.

    Returns:
        tuple: (is_installed, prolog_path)
    """
    try:
        # Try to find swipl in PATH
        prolog_path = shutil.which("swipl")
        if prolog_path:
            # Verify it works by running a simple command
            result = subprocess.run(
                [prolog_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"SWI-Prolog found at: {prolog_path}")
                return True, prolog_path

        # Try common installation paths if not in PATH
        common_paths = [
            "/usr/bin/swipl",
            "/usr/local/bin/swipl",
            "/opt/swi-prolog/bin/swipl",
            Path.home() / ".local/bin/swipl",
        ]

        for path in common_paths:
            if Path(path).exists():
                try:
                    result = subprocess.run(
                        [str(path), "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"SWI-Prolog found at: {path}")
                        return True, str(path)
                except Exception:
                    continue

        logger.warning("SWI-Prolog not found. Install with: sudo apt-get install swi-prolog")
        return False, None

    except Exception as e:
        logger.error(f"Error checking for SWI-Prolog: {e}")
        return False, None


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

@dataclass
class SemanticQuery:
    """Represents a semantic query in Prolog format"""
    query: str
    variables: list[str]
    expected_bindings: dict[str, Any]

class MontagueGrammarRules:
    """
    Montague Grammar rules implemented in Prolog

    Compositional semantics: meaning = f(syntactic_parts, semantic_rules)
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

    def get_rules_by_category(self, category: str) -> list[PrologRule]:
        """Get rules by source category"""
        return [rule for rule in self.core_rules if category in rule.source]

class PrologSemanticProcessor(AbstractSemanticProcessor):
    """
    Superior semantic processor using Prolog + Montague Grammar

    Implements formal logic reasoning instead of statistical approximation.
    Perfect for compositional semantics and logical inference.

    Note: Requires SWI-Prolog to be installed on the system for full functionality.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.grammar_rules = MontagueGrammarRules()
        self.knowledge_base = []
        self.prolog_available = False
        self.prolog_path = None

        # Check for Prolog availability
        self._check_prolog_availability()
        if self.prolog_available:
            self._initialize_knowledge_base()

    def _check_prolog_availability(self):
        """Check if Prolog is available and functional."""
        if not JANUS_AVAILABLE:
            logger.warning("janus_swi Python package not available")
            self.prolog_available = False
            return

        # Check for SWI-Prolog installation
        is_installed, prolog_path = check_prolog_installation()
        if not is_installed:
            logger.warning("SWI-Prolog not installed on system")
            self.prolog_available = False
            return

        self.prolog_path = prolog_path

        # Try to initialize Janus
        try:
            if janus is not None:
                # Test basic Prolog functionality
                result = list(janus.query("true"))
                if result is not None:
                    self.prolog_available = True
                    logger.info("Prolog functionality initialized successfully")
                else:
                    self.prolog_available = False
                    logger.warning("Prolog query test failed")
            else:
                self.prolog_available = False
                logger.warning("janus_swi is not available, cannot initialize Prolog functionality")
        except Exception as e:
            self.prolog_available = False
            logger.error(f"Failed to initialize Prolog: {e}")

    def _initialize_knowledge_base(self):
        """Initialize Prolog knowledge base with Montague Grammar rules"""
        if not self.prolog_available:
            raise SemanticProcessingError("Cannot initialize knowledge base: Prolog not available")

        try:
            # Load core Montague Grammar rules
            for rule in self.grammar_rules.core_rules:
                prolog_rule = rule.to_prolog_syntax()
                if janus is not None:
                    janus.query_once(f"assertz(({prolog_rule}))")
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
                if janus is not None:
                    janus.query_once(f"assertz({fact})")
                self.knowledge_base.append(fact)

            logger.info(f"Initialized Prolog knowledge base with {len(self.knowledge_base)} facts/rules")

        except Exception as e:
            raise SemanticProcessingError(f"Failed to initialize Prolog knowledge base: {e}") from e

    async def parse_semantics(self, text: str) -> SemanticResult:
        """
        Parse natural language into formal logical representation using Prolog

        Applies Montague Grammar principles for compositional semantics

        Raises:
            SemanticProcessingError: If Prolog is not available or parsing fails
        """
        if not self.prolog_available:
            raise SemanticProcessingError(
                "Prolog not available. Install SWI-Prolog: sudo apt-get install swi-prolog"
            )

        start_time = datetime.now()

        try:
            # Step 1: Tokenize and identify semantic components
            semantic_components = await self._extract_semantic_components(text)

            # Step 2: Apply Montague Grammar rules for composition
            logical_form = await self._compose_semantics(semantic_components)

            # Step 3: Generate Prolog representation
            prolog_facts = await self._generate_prolog_facts(logical_form, text)

            processing_time = (datetime.now() - start_time).total_seconds()

            result = SemanticResult(
                success=True,
                data={
                    "logical_form": logical_form,
                    "prolog_facts": prolog_facts,
                    "semantic_components": semantic_components
                },
                metadata={
                    "method": "prolog_montague",
                    "rule_count": len(self.grammar_rules.core_rules),
                    "components_found": len(semantic_components)
                },
                processing_time=processing_time,
                timestamp=start_time
            )

            self.update_stats(processing_time, True)
            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.update_stats(processing_time, False)

            raise SemanticProcessingError(f"Semantic parsing failed: {e}") from e

    async def extract_relations(self, semantic_data: Any) -> list[tuple[str, str, str]]:
        """
        Extract entity relationships using Prolog inference

        Uses formal logic to identify subject-predicate-object triples

        Raises:
            SemanticProcessingError: If Prolog is not available or extraction fails
        """
        if not self.prolog_available:
            raise SemanticProcessingError(
                "Cannot extract relations: Prolog not available. "
                "Install SWI-Prolog: sudo apt-get install swi-prolog"
            )

        try:
            prolog_facts = semantic_data.get("prolog_facts", [])

            # Add facts to Prolog knowledge base temporarily
            for fact in prolog_facts:
                if janus is not None:
                    janus.query_once(f"assertz({fact})")

            # Query for relationships using Prolog
            relations = []

            # Query for direct relations
            if janus is not None:
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

            return relations

        except Exception as e:
            raise SemanticProcessingError(f"Relation extraction failed: {e}") from e

    async def validate_logic(self, logical_form: LogicalForm) -> bool:
        """
        Validate logical consistency using Prolog reasoning

        Checks for contradictions and logical soundness

        Raises:
            SemanticProcessingError: If Prolog is not available or validation fails
        """
        if not self.prolog_available:
            raise SemanticProcessingError(
                "Cannot validate logic: Prolog not available. "
                "Install SWI-Prolog: sudo apt-get install swi-prolog"
            )

        try:
            # Convert logical form to Prolog format
            prolog_expression = self._convert_to_prolog(logical_form.expression)

            # Check for basic syntax validity
            if not self._is_valid_prolog_syntax(prolog_expression):
                return False

            # Attempt to assert and query the expression
            test_rule = f"test_rule :- {prolog_expression}."
            if janus is not None:
                janus.query_once(f"assertz(({test_rule}))")
            else:
                raise SemanticProcessingError("Cannot validate logic: janus_swi is not available.")

            # Try to query it - if it fails, there's a logical issue
            results = list(janus.query("test_rule")) if janus is not None else []

            # Clean up the test rule
            if janus is not None:
                janus.query_once(f"retract(({test_rule}))")

            return len(results) > 0

        except Exception as e:
            raise SemanticProcessingError(f"Logic validation failed: {e}") from e

    async def _extract_semantic_components(self, text: str) -> list[dict[str, Any]]:
        """Extract semantic components from text"""
        # Simplified component extraction
        # In a full implementation, this would use sophisticated NLP
        words = text.lower().split()
        components = []

        for word in words:
            component = {
                "word": word,
                "type": self._classify_word_type(word),
                "semantic_role": self._determine_semantic_role(word)
            }
            components.append(component)

        return components

    async def _compose_semantics(self, components: list[dict[str, Any]]) -> LogicalForm:
        """Apply Montague Grammar composition rules"""
        # Simplified composition - in full implementation, use proper Montague rules
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

    def _classify_word_type(self, word: str) -> str:
        """Classify word type for semantic analysis"""
        # Simplified classification
        if word in ["every", "some", "all", "none", "most"]:
            return "quantifier"
        elif word in ["is", "has", "does", "can", "will", "relates", "connects"]:
            return "predicate"
        else:
            return "entity"

    def _determine_semantic_role(self, word: str) -> str:
        """Determine semantic role in composition"""
        # Simplified role assignment
        return "content"

    def add_domain_rules(self, domain: str, rules: list[PrologRule]):
        """
        Add domain-specific rules to the knowledge base

        Args:
            domain: Domain name for the rules
            rules: List of PrologRule objects to add

        Raises:
            SemanticProcessingError: If Prolog is not available
        """
        if not self.prolog_available:
            raise SemanticProcessingError(
                "Cannot add domain rules: Prolog not available. "
                "Install SWI-Prolog: sudo apt-get install swi-prolog"
            )

        for rule in rules:
            prolog_rule = rule.to_prolog_syntax()
            if janus is not None:
                janus.query_once(f"assertz(({prolog_rule}))")
            self.knowledge_base.append(prolog_rule)

        logger.info(f"Added {len(rules)} rules for domain: {domain}")

    def query_knowledge_base(self, query: str) -> list[dict[str, Any]]:
        """
        Query the Prolog knowledge base directly

        Args:
            query: Prolog query string

        Returns:
            List of solution dictionaries

        Raises:
            SemanticProcessingError: If Prolog is not available
        """
        if not self.prolog_available:
            raise SemanticProcessingError(
                "Cannot query knowledge base: Prolog not available. "
                "Install SWI-Prolog: sudo apt-get install swi-prolog"
            )

        results = []
        if janus is None:
            raise SemanticProcessingError(
                "Cannot query knowledge base: janus_swi is not available."
            )
        for solution in janus.query(query):
            results.append(dict(solution))
        return results

    def _convert_to_prolog(self, expression: str) -> str:
        """
        Convert logical form expression to Prolog syntax

        Args:
            expression: Logical expression to convert

        Returns:
            Prolog-formatted expression
        """
        # Basic conversion - in a full implementation, this would handle
        # complex logical forms and proper Prolog syntax transformation
        prolog_expr = expression.lower()
        prolog_expr = prolog_expr.replace(" and ", ", ")
        prolog_expr = prolog_expr.replace(" or ", "; ")
        return prolog_expr

    def _is_valid_prolog_syntax(self, expression: str) -> bool:
        """
        Check if expression has valid Prolog syntax

        Args:
            expression: Expression to validate

        Returns:
            True if syntax appears valid
        """
        # Basic syntax validation
        if not expression:
            return False

        # Check for balanced parentheses
        paren_count = 0
        for char in expression:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if paren_count < 0:
                return False
        return paren_count == 0

    async def query_compositional(self, query_text: str, max_results: int = 10) -> list[LogicalResult]:
        """
        Execute compositional semantic query using Prolog

        Args:
            query_text: Natural language query
            max_results: Maximum results to return

        Returns:
            List of LogicalResult objects

        Raises:
            SemanticProcessingError: If Prolog is not available
        """
        if not self.prolog_available:
            raise SemanticProcessingError(
                "Cannot execute compositional query: Prolog not available. "
                "Install SWI-Prolog: sudo apt-get install swi-prolog"
            )

        # Parse the query into semantic components
        semantic_result = await self.parse_semantics(query_text)

        if not semantic_result.success:
            raise SemanticProcessingError(f"Failed to parse query: {semantic_result.error_message}")

        # Extract logical form and query Prolog
        logical_form = semantic_result.data.get("logical_form")
        results = []

        # Convert to Prolog query format
        prolog_query = self._convert_to_prolog(logical_form.expression)

        # Execute query
        if janus is None:
            raise SemanticProcessingError(
                "Cannot execute compositional query: janus_swi is not available."
            )

        start_time = datetime.now()
        for solution in janus.query(prolog_query):
            # Convert raw Prolog solution to LogicalResult
            logical_result = LogicalResult(
                data=solution,
                logical_form=str(logical_form.expression),
                confidence=0.8,  # Default confidence for Prolog results
                query_mode_used=QueryMode.LOGICAL,
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                sources_consulted=[prolog_query]
            )
            results.append(logical_result)
            if len(results) >= max_results:
                break

        return results

    async def compose_queries(self, sub_queries: list[str]) -> str:
        """
        Compose multiple sub-queries into a unified Prolog query

        Args:
            sub_queries: List of sub-query strings

        Returns:
            Composed query string

        Raises:
            SemanticProcessingError: If Prolog is not available
        """
        if not self.prolog_available:
            raise SemanticProcessingError(
                "Cannot compose queries: Prolog not available. "
                "Install SWI-Prolog: sudo apt-get install swi-prolog"
            )

        # Join sub-queries with logical AND
        composed = " , ".join(sub_queries)
        return f"({composed})"

    def get_knowledge_base_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current knowledge base

        Returns:
            Dictionary with knowledge base statistics
        """
        return {
            "prolog_available": self.prolog_available,
            "prolog_path": self.prolog_path,
            "total_rules": len(self.knowledge_base),
            "montague_rules": len(self.grammar_rules.core_rules),
            "processing_stats": self.processing_stats,
            "performance_metrics": self.get_performance_metrics()
        }
