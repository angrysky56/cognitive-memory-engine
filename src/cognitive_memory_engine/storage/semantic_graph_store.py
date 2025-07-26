"""
Semantic Graph Store Module

DuckDB-based semantic graph storage for formal logic relationships.
Complements the vector store with high-performance graph queries and reasoning.

Based on AETHELRED Elegance Toolkit using DuckDB for analytical performance.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb
else:
    try:
        import duckdb
    except ImportError:
        duckdb = None

from ..core.exceptions import CMEGraphError
from ..types import KnowledgeDomain

logger = logging.getLogger(__name__)


class SemanticGraphStore:
    """
    Semantic graph storage using DuckDB for formal logic relationships.

    Implements high-performance graph operations with SQL queries for:
    - Formal logic relationships between concepts
    - Graph traversal and path finding
    - Semantic reasoning over knowledge domains
    - Integration with existing vector similarity search
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize semantic graph store.

        Args:
            config: Graph store configuration with database_path
        """
        self.config = config
        self.database_path = Path(config.get("database_path", "cme_data/semantic_graph.db"))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection: duckdb.DuckDBPyConnection | None = None
        self.initialized = False

        if duckdb is None:
            logger.warning("DuckDB not available - using mock implementation")

        logger.info(f"Semantic graph store created with database: {self.database_path}")

    async def initialize(self) -> None:
        """Initialize DuckDB connection and create schema."""
        if self.initialized:
            return

        if duckdb is None:
            logger.warning("DuckDB not available - using mock mode")
            self.initialized = True
            return

        try:
            # Create persistent DuckDB connection
            self.connection = duckdb.connect(str(self.database_path))

            # Create semantic graph schema
            await self._create_schema()

            self.initialized = True
            logger.info("Semantic graph store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize semantic graph store: {e}")
            raise CMEGraphError(f"Initialization failed: {e}") from e

    async def _create_schema(self) -> None:
        """Create the semantic graph database schema."""
        if not self.connection:
            return

        # DuckDB schema creation using individual execute() calls
        schema_statements = [
            """
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                domain VARCHAR NOT NULL,
                description TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id VARCHAR PRIMARY KEY,
                source_concept_id VARCHAR NOT NULL,
                target_concept_id VARCHAR NOT NULL,
                relationship_type VARCHAR NOT NULL,
                weight DOUBLE DEFAULT 1.0,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_concept_id) REFERENCES concepts(concept_id),
                FOREIGN KEY (target_concept_id) REFERENCES concepts(concept_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS domains (
                domain VARCHAR PRIMARY KEY,
                description TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_concept_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_concept_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)"
        ]

        try:
            # Execute each statement individually (DuckDB pattern)
            for statement in schema_statements:
                self.connection.execute(statement.strip())
            logger.info("Semantic graph schema created successfully")
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise CMEGraphError(f"Schema creation failed: {e}") from e

    async def store_concept(self, concept_id: str, name: str, domain: KnowledgeDomain,
                          description: str = "", metadata: dict[str, Any] | None = None) -> bool:
        """
        Store a concept in the semantic graph.

        Args:
            concept_id: Unique identifier (shared with vector store)
            name: Human-readable concept name
            domain: Knowledge domain
            description: Concept description
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        if not self.initialized or not self.connection:
            logger.warning("Graph store not initialized - concept storage skipped")
            return False

        try:
            self.connection.execute("""
                INSERT OR REPLACE INTO concepts
                (concept_id, name, domain, description, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [concept_id, name, domain.value, description, json.dumps(metadata or {})])

            logger.debug(f"Stored concept: {concept_id} in domain {domain.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to store concept {concept_id}: {e}")
            return False

    async def create_relationship(self, source_id: str, target_id: str,
                                relationship_type: str, weight: float = 1.0,
                                metadata: dict[str, Any] | None = None) -> bool:
        """
        Create a semantic relationship between concepts.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship (e.g., 'contains', 'implements', 'extends')
            weight: Relationship strength (0.0 to 1.0)
            metadata: Additional relationship data

        Returns:
            True if relationship created successfully
        """
        if not self.initialized or not self.connection:
            logger.warning("Graph store not initialized - relationship creation skipped")
            return False

        try:
            relationship_id = f"{source_id}-{relationship_type}-{target_id}"

            self.connection.execute("""
                INSERT OR REPLACE INTO relationships
                (relationship_id, source_concept_id, target_concept_id,
                 relationship_type, weight, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [relationship_id, source_id, target_id, relationship_type,
                  weight, json.dumps(metadata or {})])

            logger.debug(f"Created relationship: {source_id} --{relationship_type}--> {target_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create relationship {source_id}->{target_id}: {e}")
            return False

    async def find_related_concepts(self, concept_id: str, relationship_types: list[str] | None = None,
                                  max_depth: int = 2) -> list[dict[str, Any]]:
        """
        Find concepts related to the given concept through semantic relationships.

        Args:
            concept_id: Starting concept ID
            relationship_types: Filter by relationship types (optional)
            max_depth: Maximum traversal depth

        Returns:
            List of related concepts with relationship information
        """
        if not self.initialized or not self.connection:
            logger.warning("Graph store not initialized - returning empty results")
            return []

        try:
            # Build dynamic query based on parameters
            where_clause = ""
            params = [concept_id]

            if relationship_types:
                placeholders = ",".join("?" * len(relationship_types))
                where_clause = f"AND r.relationship_type IN ({placeholders})"
                params.extend(relationship_types)

            query = f"""
                WITH RECURSIVE related_concepts AS (
                    -- Base case: direct relationships
                    SELECT
                        c.concept_id,
                        c.name,
                        c.domain,
                        c.description,
                        r.relationship_type,
                        r.weight,
                        1 as depth
                    FROM relationships r
                    JOIN concepts c ON (c.concept_id = r.target_concept_id)
                    WHERE r.source_concept_id = ?
                    {where_clause}

                    UNION ALL

                    -- Recursive case: deeper relationships
                    SELECT
                        c.concept_id,
                        c.name,
                        c.domain,
                        c.description,
                        r.relationship_type,
                        r.weight,
                        rc.depth + 1
                    FROM relationships r
                    JOIN concepts c ON (c.concept_id = r.target_concept_id)
                    JOIN related_concepts rc ON (r.source_concept_id = rc.concept_id)
                    WHERE rc.depth < ?
                    {where_clause.replace('r.source_concept_id = ?', 'r.source_concept_id = rc.concept_id')}
                )
                SELECT DISTINCT * FROM related_concepts
                ORDER BY depth, weight DESC
            """

            # Add max_depth parameter
            params.append(str(max_depth))

            results = self.connection.execute(query, params).fetchall()

            return [
                {
                    "concept_id": row[0],
                    "name": row[1],
                    "domain": row[2],
                    "description": row[3],
                    "relationship_type": row[4],
                    "weight": row[5],
                    "depth": row[6]
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Failed to find related concepts for {concept_id}: {e}")
            return []

    async def get_domain_concepts(self, domain: KnowledgeDomain, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get all concepts in a specific knowledge domain.

        Args:
            domain: Knowledge domain to query
            limit: Maximum number of concepts to return

        Returns:
            List of concepts in the domain
        """
        if not self.initialized or not self.connection:
            logger.warning("Graph store not initialized - returning empty results")
            return []

        try:
            results = self.connection.execute("""
                SELECT concept_id, name, description, metadata, created_at
                FROM concepts
                WHERE domain = ?
                ORDER BY name
                LIMIT ?
            """, [domain.value, limit]).fetchall()

            return [
                {
                    "concept_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "metadata": json.loads(row[3]) if row[3] else {},
                    "created_at": row[4]
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Failed to get domain concepts for {domain.value}: {e}")
            return []

    def analyze_graph_metrics(self) -> dict[str, Any]:
        """
        Analyze semantic graph structure and return metrics.

        Returns:
            Dictionary containing graph analysis metrics
        """
        if not self.initialized or not self.connection:
            logger.warning("Graph store not initialized - returning empty metrics")
            return {}

        try:
            # Get basic counts using DuckDB scalar query pattern
            concept_count = self.connection.execute("SELECT COUNT(*) FROM concepts").fetchall()[0][0]
            relationship_count = self.connection.execute("SELECT COUNT(*) FROM relationships").fetchall()[0][0]

            # Get domain distribution
            domain_stats = self.connection.execute("""
                SELECT domain, COUNT(*) as concept_count
                FROM concepts
                GROUP BY domain
                ORDER BY concept_count DESC
            """).fetchall()

            # Get relationship type distribution
            relationship_stats = self.connection.execute("""
                SELECT relationship_type, COUNT(*) as relationship_count
                FROM relationships
                GROUP BY relationship_type
                ORDER BY relationship_count DESC
            """).fetchall()

            return {
                "total_concepts": concept_count,
                "total_relationships": relationship_count,
                "domain_distribution": {row[0]: row[1] for row in domain_stats},
                "relationship_types": {row[0]: row[1] for row in relationship_stats},
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to analyze graph metrics: {e}")
            return {"error": str(e)}

    async def search_concepts(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for concepts matching the query text.

        Args:
            query_text: Text to search for
            limit: Maximum number of results

        Returns:
            List of concept dictionaries with relevance scores
        """
        if not self.initialized or not self.connection:
            logger.warning("Graph store not initialized - returning empty results")
            return []

        try:
            # Simple text search implementation
            results = self.connection.execute("""
                SELECT concept_id, name, domain, description,
                       1.0 as relevance_score
                FROM concepts
                WHERE LOWER(name) LIKE LOWER(?)
                   OR LOWER(description) LIKE LOWER(?)
                ORDER BY name
                LIMIT ?
            """, [f"%{query_text}%", f"%{query_text}%", limit]).fetchall()

            return [
                {
                    "concept_id": row[0],
                    "name": row[1],
                    "domain": row[2],
                    "description": row[3],
                    "relevance_score": row[4]
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Failed to search concepts: {e}")
            return []

    async def traverse_relationships(self, start_nodes: list[str],
                                   query_pattern: str) -> list[dict[str, Any]]:
        """
        Traverse relationships from starting nodes following a pattern.

        Args:
            start_nodes: List of starting concept IDs
            query_pattern: Pattern to follow (relationship type)

        Returns:
            List of relationship traversal results
        """
        if not self.initialized or not self.connection:
            logger.warning("Graph store not initialized - returning empty results")
            return []

        try:
            placeholders = ",".join("?" * len(start_nodes))
            results = self.connection.execute(f"""
                SELECT r.source_concept_id, r.target_concept_id,
                       r.relationship_type, r.weight
                FROM relationships r
                WHERE r.source_concept_id IN ({placeholders})
                  AND r.relationship_type = ?
                ORDER BY r.weight DESC
            """, start_nodes + [query_pattern]).fetchall()

            return [
                {
                    "source_concept_id": row[0],
                    "target_concept_id": row[1],
                    "relationship_type": row[2],
                    "weight": row[3]
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Failed to traverse relationships: {e}")
            return []

    async def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Semantic graph store connection closed")
