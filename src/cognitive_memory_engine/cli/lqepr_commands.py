"""
CLI Interface for LQEPR Unified Query System

Elegant command-line interface using typer + rich for Phase 3B-2.
Provides interactive query building and beautiful result visualization.

Following AETHELRED Elegance Toolkit:
- typer: Modern, elegant CLI framework
- rich: Beautiful terminal formatting
- async support: Responsive interface
"""

import asyncio
from pathlib import Path
from typing import List, Optional

try:
    import typer
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError:
    # Graceful fallback without rich/typer
    typer = None
    Console = None
    rprint = print

from ..semantic.logical_query_processor import LogicalQueryProcessor, LQEPRResult
from ..types import QueryMode

# Global console for rich formatting
console = Console() if Console else None

# CLI app instance
app = typer.Typer(
    name="lqepr",
    help="LQEPR (Logical Query Enhanced Pattern Retrieval) CLI",
    rich_markup_mode="rich" if typer else None
) if typer else None


def init_query_processor() -> LogicalQueryProcessor:
    """Initialize LQEPR query processor with available components"""
    # Import here to avoid circular dependencies
    from ..core.engine import CognitiveMemoryEngine

    # Initialize engine and get components
    engine = CognitiveMemoryEngine()

    # Try to get components from engine
    prolog_processor = getattr(engine, 'prolog_processor', None)
    semantic_graph_store = getattr(engine, 'semantic_graph_store', None)
    vector_store = getattr(engine, 'vector_store', None)

    return LogicalQueryProcessor(
        prolog_processor=prolog_processor,
        semantic_graph_store=semantic_graph_store,
        vector_store=vector_store
    )


@app.command("query") if app else lambda: None
def query_command(
    query_text: str = typer.Argument(..., help="Natural language query text"),
    modes: Optional[List[str]] = typer.Option(
        None, "--mode", "-m",
        help="Query modes: logical, graph, vector (default: all available)"
    ),
    max_results: int = typer.Option(10, "--max", "-n", help="Maximum results per mode"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed results"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress display"),
    timeout: float = typer.Option(30.0, "--timeout", "-t", help="Query timeout in seconds")
):
    """
    Execute unified LQEPR query across multiple semantic backends.

    Examples:
        lqepr query "What is photosynthesis?"
        lqepr query "Find concepts related to memory" --mode logical --mode graph
        lqepr query "Neural networks and learning" --detailed --max 20
    """
    # Parse query modes
    query_modes = []
    if modes:
        mode_map = {
            'logical': QueryMode.LOGICAL,
            'graph': QueryMode.GRAPH,
            'vector': QueryMode.VECTOR
        }
        query_modes = [mode_map[m] for m in modes if m in mode_map]

    # Execute query
    result = asyncio.run(_execute_query(
        query_text=query_text,
        modes=query_modes or None,
        max_results=max_results,
        timeout=timeout,
        show_progress=not no_progress
    ))

    # Display results
    processor = init_query_processor()
    processor.display_results(result, detailed=detailed)


@app.command("interactive") if app else lambda: None
def interactive_command():
    """
    Interactive LQEPR query session with guided prompts.

    Provides a conversational interface for building and executing queries.
    """
    if not console:
        print("Interactive mode requires rich library")
        return

    console.print(Panel(
        "[bold blue]LQEPR Interactive Query Session[/bold blue]\n"
        "Build sophisticated queries using guided prompts.\n"
        "Type 'exit' or 'quit' to end session.",
        title="Interactive Mode"
    ))

    processor = init_query_processor()

    # Show available modes
    stats = processor.get_stats()
    available_modes = stats['available_modes']

    console.print(f"[green]Available query modes:[/green] {', '.join(available_modes)}")
    console.print()

    while True:
        try:
            # Get query from user
            query_text = Prompt.ask("[bold cyan]Enter your query[/bold cyan]")

            if query_text.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break

            # Get query preferences
            use_all_modes = Confirm.ask("Use all available query modes?", default=True)

            modes = None
            if not use_all_modes:
                selected_modes = []
                for mode in available_modes:
                    if Confirm.ask(f"Include {mode} mode?"):
                        mode_map = {
                            'logical': QueryMode.LOGICAL,
                            'graph': QueryMode.GRAPH,
                            'vector': QueryMode.VECTOR
                        }
                        selected_modes.append(mode_map[mode])
                modes = selected_modes

            max_results = int(Prompt.ask("Maximum results per mode", default="10"))
            detailed = Confirm.ask("Show detailed results?", default=False)

            # Execute query
            result = asyncio.run(_execute_query(
                query_text=query_text,
                modes=modes,
                max_results=max_results,
                show_progress=True
            ))

            # Display results
            processor.display_results(result, detailed=detailed)
            console.print()

            # Ask if user wants to continue
            if not Confirm.ask("Execute another query?", default=True):
                break

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command("stats") if app else lambda: None
def stats_command():
    """Display LQEPR system statistics and performance metrics."""
    processor = init_query_processor()
    stats = processor.get_stats()

    if not console:
        print(f"LQEPR Statistics: {stats}")
        return

    # Create rich table for statistics
    table = Table(title="LQEPR System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    table.add_row("Total Queries", str(stats['total_queries']))
    table.add_row("Average Execution Time", f"{stats['avg_execution_time']:.3f}s")
    table.add_row("Cache Hits", str(stats['cache_hits']))
    table.add_row("Cache Size", str(stats['cache_size']))
    table.add_row("Available Modes", ", ".join(stats['available_modes']))

    # Mode usage breakdown
    for mode, count in stats['mode_usage'].items():
        table.add_row(f"{mode.value.title()} Mode Usage", str(count))

    console.print(table)


@app.command("test") if app else lambda: None
def test_command(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose test output")
):
    """
    Test LQEPR system components and verify functionality.

    Runs comprehensive tests of all query modes and reports results.
    """
    asyncio.run(_run_tests(verbose=verbose))


async def _execute_query(query_text: str,
                        modes: Optional[List[QueryMode]] = None,
                        max_results: int = 10,
                        timeout: float = 30.0,
                        show_progress: bool = True) -> LQEPRResult:
    """Execute LQEPR query with error handling"""
    try:
        processor = init_query_processor()

        result = await processor.unified_query(
            query_text=query_text,
            modes=modes,
            max_results=max_results,
            timeout=timeout,
            show_progress=show_progress
        )

        return result

    except Exception as e:
        if console:
            console.print(f"[red]Query execution failed: {e}[/red]")
        else:
            print(f"Query execution failed: {e}")

        # Return empty result
        return LQEPRResult(
            query_text=query_text,
            query_mode=QueryMode.UNIFIED,
            logical_results=[],
            graph_results=[],
            vector_results=[],
            unified_score=0.0,
            confidence=0.0,
            execution_time=0.0,
            query_metadata={'error': str(e)}
        )


async def _run_tests(verbose: bool = False):
    """Run LQEPR system tests"""
    if not console:
        print("Running LQEPR tests...")
    else:
        console.print(Panel(
            "[bold blue]LQEPR System Tests[/bold blue]\n"
            "Testing all components and query modes...",
            title="System Verification"
        ))

    processor = init_query_processor()
    test_queries = [
        "What is artificial intelligence?",
        "Find concepts related to memory and learning",
        "How do neural networks work?",
        "Explain semantic reasoning and logic"
    ]

    results = []

    for i, query in enumerate(test_queries, 1):
        if console:
            console.print(f"[cyan]Test {i}:[/cyan] {query}")
        else:
            print(f"Test {i}: {query}")

        try:
            result = await processor.unified_query(
                query_text=query,
                max_results=5,
                show_progress=False
            )

            success = (result.logical_results or result.graph_results or result.vector_results)
            results.append(success)

            if verbose:
                processor.display_results(result, detailed=True)
            elif console:
                status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
                console.print(f"  {status} - Score: {result.unified_score:.3f}")
            else:
                status = "PASS" if success else "FAIL"
                print(f"  {status} - Score: {result.unified_score:.3f}")

        except Exception as e:
            results.append(False)
            if console:
                console.print(f"  [red]✗ ERROR: {e}[/red]")
            else:
                print(f"  ERROR: {e}")

    # Summary
    passed = sum(results)
    total = len(results)

    if console:
        color = "green" if passed == total else "yellow" if passed > 0 else "red"
        console.print(f"\n[{color}]Tests passed: {passed}/{total}[/{color}]")

        if passed == total:
            console.print("[green]🎉 All LQEPR components working correctly![/green]")
        elif passed > 0:
            console.print("[yellow]⚠️  Some components may need attention[/yellow]")
        else:
            console.print("[red]❌ LQEPR system needs debugging[/red]")
    else:
        print(f"Tests passed: {passed}/{total}")


def main():
    """Main CLI entry point"""
    if not app:
        print("CLI requires typer library. Install with: pip install typer rich")
        return

    app()


if __name__ == "__main__":
    main()
