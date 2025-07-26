"""
Document Knowledge Builder

Builds formal knowledge structures from documents using RTM principles,
implementing the document analysis component of Track 2.

Extracts hierarchical concept structures, creates DocumentRTM objects,
and organizes knowledge for efficient retrieval.
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Any

from ..llm_providers import LLMProviderFactory
from ..storage.document_store import DocumentStore
from ..types import DocumentRTM, KnowledgeConcept, KnowledgeDomain

logger = logging.getLogger(__name__)


class DocumentKnowledgeBuilder:
    """
    Builds structured knowledge representations from formal documents.

    Uses LLM analysis to:
    - Extract hierarchical concept structures
    - Identify key components and relationships
    - Create formal knowledge RTMs (not conversation narratives)
    - Organize concepts for efficient retrieval
    """

    def __init__(self,
                 cloud_config,
                 document_store: DocumentStore,
                 max_branching_factor: int = 4,
                 max_depth: int = 6):
        """
        Initialize document knowledge builder.

        Args:
            cloud_config: Cloud provider configuration
            document_store: Document storage system
            max_branching_factor: Maximum children per concept (RTM K parameter)
            max_depth: Maximum depth of knowledge hierarchy
        """
        self.cloud_config = cloud_config
        self.document_store = document_store
        self.max_branching_factor = max_branching_factor
        self.max_depth = max_depth

        # Create LLM provider using factory (same pattern as narrative_tree_builder)
        self.provider = LLMProviderFactory.create(cloud_config)

        logger.info("Document Knowledge Builder initialized")

    async def build_document_rtm(self,
                               document_content: str,
                               root_concept: str,
                               domain: KnowledgeDomain,
                               metadata: dict[str, Any] | None = None) -> DocumentRTM:
        """
        Build a formal DocumentRTM from document content.

        Args:
            document_content: Raw document text
            root_concept: Main concept name (e.g., "SAPE")
            domain: Knowledge domain
            metadata: Optional source metadata

        Returns:
            DocumentRTM with hierarchical concept structure
        """
        try:
            logger.info(f"Building document RTM for concept '{root_concept}' in domain {domain.value}")

            # Step 1: Extract document structure and main components
            structure_analysis = await self._analyze_document_structure(
                document_content, root_concept
            )

            # Step 2: Create root concept
            root_concept_obj = await self._create_root_concept(
                root_concept, structure_analysis, domain
            )

            # Step 3: Build hierarchical concept tree
            concepts = {root_concept_obj.concept_id: root_concept_obj}
            await self._build_concept_hierarchy(
                document_content,
                root_concept_obj,
                structure_analysis,
                concepts,
                depth=1
            )

            # Step 4: Create DocumentRTM
            document = DocumentRTM(
                doc_id=str(uuid.uuid4()),
                title=f"{root_concept} - {domain.value.replace('_', ' ').title()}",
                root_concept=root_concept,
                domain=domain,
                root_concept_id=root_concept_obj.concept_id,
                concepts=concepts,
                source_content=document_content,
                source_metadata=metadata or {},
                max_branching_factor=self.max_branching_factor,
                max_depth=self.max_depth,
                total_concepts=len(concepts),
                compression_ratio=len(document_content) / max(len(concepts), 1),
                created=datetime.now(),
                last_accessed=datetime.now()
            )

            logger.info(f"Built DocumentRTM with {len(concepts)} concepts")
            return document

        except Exception as e:
            logger.error(f"Failed to build DocumentRTM for {root_concept}: {e}")
            raise

    async def _analyze_document_structure(self, content: str, root_concept: str) -> dict[str, Any]:
        """
        Analyze document to extract high-level structure and components.

        Args:
            content: Document content
            root_concept: Main concept name

        Returns:
            Structure analysis with main components and hierarchy
        """
        prompt = f"""
        Analyze this document about {root_concept} and extract its hierarchical structure.

        Document content:
        {content[:3000]}...

        Please provide:
        1. Main components (up to {self.max_branching_factor} key parts)
        2. Hierarchical relationships between components
        3. Key concepts and their definitions
        4. Implementation details or examples

        Format as JSON with:
        {{
            "main_components": ["component1", "component2", ...],
            "component_descriptions": {{"component1": "description", ...}},
            "hierarchical_structure": {{"component1": ["subcomponent1", "subcomponent2"], ...}},
            "key_concepts": {{"concept_name": "definition", ...}},
            "implementation_details": {{"component1": "details", ...}}
        }}
        """

        try:
            response = await self.provider.generate(prompt)

            # Extract JSON from response
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                logger.info(f"Extracted structure with {len(analysis.get('main_components', []))} main components")
                return analysis
            else:
                logger.warning("Could not parse structure analysis JSON, using fallback")
                return self._create_fallback_structure(content, root_concept)

        except Exception as e:
            logger.warning(f"Structure analysis failed: {e}, using fallback")
            return self._create_fallback_structure(content, root_concept)

    def _create_fallback_structure(self, content: str, root_concept: str) -> dict[str, Any]:
        """Create basic structure analysis when LLM parsing fails."""
        # Extract potential components by looking for headers, caps words, etc.
        lines = content.split('\n')
        components = []

        for line in lines[:50]:  # First 50 lines
            line = line.strip()
            # Look for capitalized words that might be components
            words = re.findall(r'\b[A-Z]{2,}\b', line)
            components.extend(words)

        # Take unique components, limit to branching factor
        unique_components = list(set(components))[:self.max_branching_factor]

        return {
            "main_components": unique_components,
            "component_descriptions": dict.fromkeys(unique_components, f"Component of {root_concept}"),
            "hierarchical_structure": {comp: [] for comp in unique_components},
            "key_concepts": {root_concept: "Main concept from document"},
            "implementation_details": {}
        }

    async def _create_root_concept(self,
                                 concept_name: str,
                                 structure_analysis: dict[str, Any],
                                 domain: KnowledgeDomain) -> KnowledgeConcept:
        """Create the root concept object."""

        # Generate description for root concept
        description = structure_analysis.get("key_concepts", {}).get(concept_name,
                                                                   f"Root concept for {concept_name}")

        # Extract main content about the concept
        main_components = structure_analysis.get("main_components", [])
        content = f"Main concept: {concept_name}\n"
        content += f"Key components: {', '.join(main_components)}\n"
        content += f"Description: {description}"

        root_concept = KnowledgeConcept(
            name=concept_name,
            description=description,
            content=content,
            structured_data={
                "main_components": main_components,
                "structure_analysis": structure_analysis
            },
            domain=domain,
            salience_score=1.0,  # Root concept has highest salience
            confidence_score=0.9,
            created=datetime.now(),
            last_updated=datetime.now()
        )

        logger.info(f"Created root concept '{concept_name}' with {len(main_components)} components")
        return root_concept

    async def _build_concept_hierarchy(self,
                                     document_content: str,
                                     parent_concept: KnowledgeConcept,
                                     structure_analysis: dict[str, Any],
                                     concepts: dict[str, KnowledgeConcept],
                                     depth: int) -> None:
        """
        Recursively build concept hierarchy using RTM principles.

        Args:
            document_content: Full document content
            parent_concept: Parent concept to build children for
            structure_analysis: Document structure analysis
            concepts: Dictionary to store all concepts
            depth: Current depth in hierarchy
        """
        if depth >= self.max_depth:
            return

        # Get child components for this concept
        child_component_names = self._get_child_components(parent_concept, structure_analysis)

        if not child_component_names:
            return

        logger.info(f"Building {len(child_component_names)} child concepts for {parent_concept.name} at depth {depth}")

        for component_name in child_component_names:
            # Create child concept
            child_concept = await self._create_child_concept(
                component_name,
                parent_concept,
                document_content,
                structure_analysis
            )

            # Add to concepts dictionary
            concepts[child_concept.concept_id] = child_concept

            # Link parent and child
            parent_concept.child_concept_ids.append(child_concept.concept_id)
            child_concept.parent_concept_id = parent_concept.concept_id

            # Recursively build children of this child
            await self._build_concept_hierarchy(
                document_content,
                child_concept,
                structure_analysis,
                concepts,
                depth + 1
            )

    def _get_child_components(self,
                            parent_concept: KnowledgeConcept,
                            structure_analysis: dict[str, Any]) -> list[str]:
        """Get child component names for a concept."""

        hierarchical_structure = structure_analysis.get("hierarchical_structure", {})

        # If parent is root, get main components
        if parent_concept.parent_concept_id is None:
            return structure_analysis.get("main_components", [])[:self.max_branching_factor]

        # Otherwise, look for sub-components in hierarchical structure
        parent_name = parent_concept.name
        return hierarchical_structure.get(parent_name, [])[:self.max_branching_factor]

    async def _create_child_concept(self,
                                  component_name: str,
                                  parent_concept: KnowledgeConcept,
                                  document_content: str,
                                  structure_analysis: dict[str, Any]) -> KnowledgeConcept:
        """Create a child concept with detailed analysis."""

        # Get description from structure analysis
        descriptions = structure_analysis.get("component_descriptions", {})
        description = descriptions.get(component_name, f"Component of {parent_concept.name}")

        # Extract relevant content for this component
        relevant_content = await self._extract_component_content(
            component_name, document_content
        )

        # Calculate salience based on depth and content
        salience = max(0.1, 1.0 - (len(parent_concept.child_concept_ids) * 0.1))

        child_concept = KnowledgeConcept(
            name=component_name,
            description=description,
            parent_concept_id=parent_concept.concept_id,
            content=relevant_content,
            structured_data={
                "parent_concept": parent_concept.name,
                "component_type": "formal_component"
            },
            domain=parent_concept.domain,
            salience_score=salience,
            confidence_score=0.8,
            created=datetime.now(),
            last_updated=datetime.now()
        )

        logger.debug(f"Created child concept '{component_name}' under '{parent_concept.name}'")
        return child_concept

    async def _extract_component_content(self, component_name: str, document_content: str) -> str:
        """Extract relevant content for a specific component."""

        # Simple extraction: look for sections mentioning the component
        lines = document_content.split('\n')
        relevant_lines = []

        component_lower = component_name.lower()

        for i, line in enumerate(lines):
            if component_lower in line.lower():
                # Include this line and some context
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                relevant_lines.extend(lines[start:end])

        # Take first 500 characters
        content = '\n'.join(relevant_lines)[:500]

        if not content:
            content = f"Component: {component_name}\nNo specific content found in document."

        return content

    async def store_document_knowledge(self,
                                     document_content: str,
                                     root_concept: str,
                                     domain: KnowledgeDomain,
                                     metadata: dict[str, Any] | None = None) -> str:
        """
        Build and store complete document knowledge structure.

        Args:
            document_content: Raw document content
            root_concept: Main concept name
            domain: Knowledge domain
            metadata: Optional metadata

        Returns:
            Document ID of stored knowledge
        """
        try:
            # Build DocumentRTM
            document = await self.build_document_rtm(
                document_content, root_concept, domain, metadata
            )

            # Store in document store
            success = await self.document_store.store_document(document)

            if success:
                logger.info(f"Successfully stored document knowledge for {root_concept}")
                return document.doc_id
            else:
                raise Exception("Failed to store document in document store")

        except Exception as e:
            logger.error(f"Failed to store document knowledge for {root_concept}: {e}")
            raise
