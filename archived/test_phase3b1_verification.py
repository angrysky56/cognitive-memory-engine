#!/usr/bin/env python3
"""
Phase 3B-1 Verification Test

Tests DuckDB Semantic Graph Store integration following AETHELRED methodology.
Uses Elegance Toolkit (typer + rich) for clean CLI testing interface.
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

# Import the new semantic graph store
from src.cognitive_memory_engine.storage.semantic_graph_store import SemanticGraphStore
from src.cognitive_memory_engine.types import KnowledgeDomain

console = Console()
app = typer.Typer(help="Verify Phase 3B-1 DuckDB Semantic Graph Store implementation")


@app.command()
def test_integration():
    """Test semantic graph store integration with sample data."""
    console.print(Panel.fit("🧠 Phase 3B-1 Verification Test", style="bold blue"))

    async def run_test():
        # Initialize semantic graph store
        config = {"database_path": "test_data/test_semantic_graph.db"}
        graph_store = SemanticGraphStore(config)

        console.print("1. Initializing DuckDB semantic graph store...")
        await graph_store.initialize()

        if not graph_store.initialized:
            console.print("❌ Failed to initialize graph store", style="red")
            return

        console.print("✅ Graph store initialized successfully", style="green")

        # Test concept storage
        console.print("2. Testing concept storage...")
        concepts = [
            ("aethelred_core", "AETHELRED Core Philosophy", KnowledgeDomain.SOFTWARE_ENGINEERING,
             "Code elegance principles focusing on clarity and simplicity"),
            ("duckdb_integration", "DuckDB Integration", KnowledgeDomain.AI_ARCHITECTURE,
             "Semantic graph store using DuckDB for formal logic relationships"),
            ("elegance_toolkit", "Elegance Toolkit", KnowledgeDomain.SOFTWARE_ENGINEERING,
             "Curated library recommendations for elegant Python development")
        ]

        for concept_id, name, domain, description in concepts:
            success = await graph_store.store_concept(concept_id, name, domain, description)
            if success:
                console.print(f"  ✅ Stored: {name}", style="green")
            else:
                console.print(f"  ❌ Failed: {name}", style="red")

        # Test relationship creation
        console.print("3. Testing semantic relationships...")
        relationships = [
            ("aethelred_core", "elegance_toolkit", "uses", 0.9),
            ("duckdb_integration", "aethelred_core", "follows", 0.8),
            ("elegance_toolkit", "duckdb_integration", "enables", 0.7)
        ]

        for source, target, rel_type, weight in relationships:
            success = await graph_store.create_relationship(source, target, rel_type, weight)
            if success:
                console.print(f"  ✅ Linked: {source} --{rel_type}--> {target}", style="green")
            else:
                console.print(f"  ❌ Failed: {source} --{rel_type}--> {target}", style="red")

        # Test graph queries
        console.print("4. Testing graph traversal...")
        related = await graph_store.find_related_concepts("aethelred_core", max_depth=2)
        console.print(f"  Found {len(related)} related concepts:", style="blue")
        for concept in related:
            console.print(f"    • {concept['name']} ({concept['relationship_type']}, depth: {concept['depth']})")

        # Test domain queries
        console.print("5. Testing domain queries...")
        se_concepts = await graph_store.get_domain_concepts(KnowledgeDomain.SOFTWARE_ENGINEERING)
        console.print(f"  Software Engineering concepts: {len(se_concepts)}", style="blue")

        # Test graph metrics
        console.print("6. Testing graph analysis...")
        metrics = graph_store.analyze_graph_metrics()
        console.print(f"  Total concepts: {metrics.get('total_concepts', 0)}", style="blue")
        console.print(f"  Total relationships: {metrics.get('total_relationships', 0)}", style="blue")

        await graph_store.close()
        console.print(Panel("✅ All tests completed successfully!", style="bold green"))

    # Run async test
    asyncio.run(run_test())


@app.command()
def cleanup():
    """Clean up test database files."""
    test_db = Path("test_data/test_semantic_graph.db")
    if test_db.exists():
        test_db.unlink()
        console.print("🧹 Test database cleaned up", style="yellow")
    else:
        console.print("ℹ️  No test database to clean", style="blue")


if __name__ == "__main__":
    app()
