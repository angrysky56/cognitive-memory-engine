#!/usr/bin/env python3
"""
Cognitive Memory Engine MCP Server

A Model Context Protocol server that provides access to the Cognitive Memory Engine
for advanced AI memory management and retrieval.
"""

import asyncio
import atexit
import json
import logging
import signal
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

import mcp.server.stdio
from mcp import types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic.networks import AnyUrl

from ..config import load_env_config
from ..core.engine import CognitiveMemoryEngine
from ..core.exceptions import CMEError
from ..task_manager import create_immediate_response, task_manager

if TYPE_CHECKING:
    from cognitive_memory_engine.mcp_server.enhanced_server_tools import EnhancedKnowledgeServerTools

# Configure logging to stderr for MCP servers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Enhanced Knowledge Tools Integration
enhanced_tools_available = False
enhanced_tools = None

def import_enhanced_tools():
    """Safely import enhanced knowledge tools."""
    global enhanced_tools_available
    try:
        from pathlib import Path
        enhanced_tools_path = Path(__file__).parent.parent.parent.parent / "enhanced_knowledge_tools"
        if enhanced_tools_path.exists():
            import sys
            sys.path.insert(0, str(enhanced_tools_path))
            from cognitive_memory_engine.mcp_server.enhanced_server_tools import EnhancedKnowledgeServerTools
            enhanced_tools_available = True
            logger.info("Enhanced knowledge tools imported successfully")
            return EnhancedKnowledgeServerTools
        else:
            logger.warning("Enhanced knowledge tools directory not found")
            return None
    except ImportError as e:
        logger.warning(f"Enhanced knowledge tools not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Error importing enhanced knowledge tools: {e}")
        return None

# Global engine instance
engine: CognitiveMemoryEngine | None = None

# Global enhanced tools instance
enhanced_tools: "EnhancedKnowledgeServerTools | None" = None

# Create MCP server
server = Server("cognitive-memory-engine")


async def initialize_engine() -> CognitiveMemoryEngine:
    """Initialize the Cognitive Memory Engine with environment configuration."""
    global engine, enhanced_tools

    if engine is None:
        try:
            logger.info("Loading configuration from environment...")
            config = load_env_config()
            logger.info(f"Using LLM model: {config.llm_model}")

            # Step 1: Create and initialize the core engine first.
            logger.info("Initializing Cognitive Memory Engine...")
            engine = CognitiveMemoryEngine(config)
            await engine.initialize()
            logger.info("Cognitive Memory Engine initialized successfully")

            # Step 2: Now that the engine exists, initialize the enhanced tools with it.
            logger.info("Initializing Enhanced Knowledge Tools...")
            enhanced_tools_class = import_enhanced_tools()
            if enhanced_tools_class and enhanced_tools_available:
                enhanced_tools = enhanced_tools_class(engine)

                # Step 3: Connect MCP tools to enhanced tools if available
                await _connect_mcp_tools_to_enhanced(enhanced_tools)

                logger.info("Enhanced knowledge tools initialized successfully")

                # Step 4: Attach the initialized tools back to the engine instance.
                engine.enhanced_knowledge_tools = enhanced_tools
            else:
                logger.warning("Enhanced tools not available, continuing without them.")

        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Memory Engine: {e}")
            # Reset globals on failure to allow for a retry
            engine = None
            enhanced_tools = None
            raise CMEError(f"Engine initialization failed: {e}") from e

    return engine


async def _connect_mcp_tools_to_enhanced(enhanced_tools_instance):
    """Connect available MCP tools to enhanced knowledge tools."""
    try:
        # Check for available MCP tools and connect them
        # These would be the actual MCP tool handlers available in the server
        available_tools = {
            "web_search": None,  # Would connect to actual web_search tool
            "web_fetch": None,   # Would connect to actual web_fetch tool
            "mcp-server-firecrawl:firecrawl_search": None,
            "mcp-server-firecrawl:firecrawl_scrape": None,
        }

        # For now, mark tools as potentially available
        # In a full implementation, this would connect actual tool handlers
        connected_count = 0
        for tool_name, handler in available_tools.items():
            if handler:  # If tool handler is available
                enhanced_tools_instance.set_mcp_tool(tool_name, handler)
                connected_count += 1

        logger.info(f"Connected {connected_count} MCP tools to enhanced knowledge system")

        # Set integration features based on available tools
        if hasattr(enhanced_tools_instance, 'integration_features'):
            enhanced_tools_instance.integration_features = {
                "fetch_enabled": connected_count > 0,
                "context_enhanced": True,  # Context enhancement is always available
                "enhanced_tools_available": True
            }

    except Exception as e:
        logger.warning(f"Failed to connect MCP tools to enhanced system: {e}")
        # Continue without enhanced tools rather than failing
        pass


async def async_cleanup():
    """Async cleanup function."""
    global engine

    logger.info("Cleaning up Cognitive Memory Engine")

    if engine:
        try:
            await engine.cleanup()
            logger.info("Engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during engine cleanup: {e}")
        finally:
            engine = None


def cleanup():
    """Clean up resources on shutdown - sync version for signal handlers."""
    global engine

    logger.info("Shutting down Cognitive Memory Engine MCP server")

    if engine:
        try:
            # For signal handlers, just set engine to None
            # The actual cleanup will happen in main()
            engine = None
            logger.info("Engine marked for cleanup")
        except Exception as e:
            logger.error(f"Error during engine cleanup: {e}")


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down gracefully")
    cleanup()
    # Don't call sys.exit() here as it interferes with asyncio
    # The main loop will handle the exit


# Register cleanup handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup)


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available memory resources."""
    return [
        types.Resource(
            uri=AnyUrl("cme://memory/conversations"),
            name="Recent Conversations",
            description="Access to recent conversation history with RTM structures",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cme://memory/documents"),
            name="Document Knowledge",
            description="Document knowledge organized by domain",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cme://memory/concepts"),
            name="Formal Concepts",
            description="Direct access to formal concepts",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cme://memory/cross_references"),
            name="Cross-References",
            description="Links between conversation and document knowledge tracks",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cme://memory/temporal"),
            name="Temporal Books",
            description="Time-organized memory books and shelves",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cme://memory/context"),
            name="Active Context",
            description="Current active workspace context",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cme://memory/narratives"),
            name="Narrative Trees",
            description="Hierarchical narrative structures from conversations",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cme://memory/vector_store"),
            name="Vector Store Details",
            description="Detailed statistics and configuration of the vector store",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read a specific memory resource."""
    try:
        cme = await initialize_engine()

        if uri == "cme://memory/conversations":
            conversations = await cme.get_recent_conversations(limit=10)
            return json.dumps(conversations, indent=2, default=str)

        elif uri == "cme://memory/documents":
            all_docs_summary = []
            if cme.document_store:
                all_docs = await cme.document_store.get_all_documents()
                for doc in all_docs:
                    all_docs_summary.append({
                        "document_id": doc.doc_id,
                        "title": doc.title,
                        "root_concept": doc.root_concept,
                        "domain": doc.domain.value,
                        "total_concepts": doc.total_concepts,
                        "created": doc.created.isoformat(),
                    })
            return json.dumps(all_docs_summary, indent=2, default=str)

        elif uri == "cme://memory/concepts":
            all_concepts_summary = []
            if cme.document_store:
                all_docs = await cme.document_store.get_all_documents()
                for doc in all_docs:
                    for _concept_id, concept in doc.concepts.items():
                        all_concepts_summary.append({
                            "concept_id": concept.concept_id,
                            "name": concept.name,
                            "description": concept.description[:150] + "..." if len(concept.description) > 150 else concept.description,
                            "document_id": doc.doc_id,
                            "domain": doc.domain.value,
                        })
            return json.dumps(all_concepts_summary, indent=2, default=str)

        elif uri == "cme://memory/cross_references":
            # TODO: Implement retrieval of cross-references
            cross_refs = {"status": "not_implemented", "message": "Retrieval of cross-references is not yet implemented."}
            return json.dumps(cross_refs, indent=2)

        elif uri == "cme://memory/narratives":
            narratives = await cme.get_narrative_summaries()
            return json.dumps(narratives, indent=2, default=str)

        elif uri == "cme://memory/temporal":
            temporal_data = await cme.get_temporal_organization()
            return json.dumps(temporal_data, indent=2, default=str)

        elif uri == "cme://memory/context":
            context = await cme.get_active_context()
            return json.dumps(context, indent=2, default=str)

        elif uri == "cme://memory/vector_store":
            stats = await cme.get_memory_stats(include_details=True)
            vector_store_details = stats.get("vector_store", {})
            return json.dumps(vector_store_details, indent=2, default=str)

        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        error_response = {
            "error": str(e),
            "uri": uri,
            "timestamp": asyncio.get_event_loop().time()
        }
        return json.dumps(error_response, indent=2)


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """list available memory tools."""
    return [
        types.Tool(
            name="store_conversation",
            description="Store a conversation in memory with narrative analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation": {
                        "type": "array",
                        "description": "Array of conversation messages",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                                "content": {"type": "string"},
                                "timestamp": {"type": "string", "format": "date-time", "description": "Optional timestamp"}
                            },
                            "required": ["role", "content"]
                        }
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context metadata",
                        "properties": {
                            "topic": {"type": "string"},
                            "participants": {"type": "array", "items": {"type": "string"}},
                            "importance": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                },
                "required": ["conversation"]
            },
        ),
        types.Tool(
            name="query_memory",
            description="Search and retrieve information from memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query for memory search"
                    },
                    "context_depth": {
                        "type": "integer",
                        "description": "RTM tree traversal depth (1-10)",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3
                    },
                    "time_scope": {
                        "type": "string",
                        "description": "Temporal scope for search",
                        "enum": ["hour", "day", "week", "month", "year", "all"],
                        "default": "week"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10
                    }
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="generate_response",
            description="Generate a contextually aware response using memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Input prompt for response generation"
                    },
                    "context_depth": {
                        "type": "integer",
                        "description": "Memory context depth to include",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3
                    },
                    "response_style": {
                        "type": "string",
                        "description": "Style of response to generate",
                        "enum": ["analytical", "conversational", "technical", "creative"],
                        "default": "conversational"
                    }
                },
                "required": ["prompt"]
            },
        ),
        types.Tool(
            name="analyze_conversation",
            description="Perform deep analysis of conversation patterns and themes",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "ID of conversation to analyze (optional, analyzes recent if not provided)"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis to perform",
                        "enum": ["narrative", "temporal", "semantic", "social", "all"],
                        "default": "all"
                    }
                },
                "required": []
            },
        ),
        types.Tool(
            name="get_memory_stats",
            description="Get statistics and insights about the memory system",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed breakdown of memory components",
                        "default": False
                    }
                },
                "required": []
            },
        ),
        types.Tool(
            name="get_rtm_tree_details",
            description="Get detailed information about a specific RTM tree",
            inputSchema={
                "type": "object",
                "properties": {
                    "tree_id": {
                        "type": "string",
                        "description": "ID of the RTM tree to retrieve details for"
                    }
                },
                "required": ["tree_id"]
            },
        ),
        types.Tool(
            name="get_vector_store_details",
            description="Get detailed statistics and configuration of the vector store",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed breakdown of vector store components",
                        "default": False
                    }
                },
                "required": []
            },
        ),
        types.Tool(
            name="get_task_status",
            description="Get the status of a background task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the task to check"
                    }
                },
                "required": ["task_id"]
            },
        ),
        types.Tool(
            name="list_tasks",
            description="List all background tasks and their status",
            inputSchema={
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "description": "Filter tasks by status (pending, running, completed, failed, all)",
                        "default": "all"
                    }
                },
                "required": []
            },
        ),
        types.Tool(
            name="store_document_knowledge",
            description="Store formal documents as structured knowledge with optional fetch integration",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_content": {
                        "type": "string",
                        "description": "The document content to store (optional if source_url provided)"
                    },
                    "source_url": {
                        "type": "string",
                        "description": "URL to fetch content from (optional if document_content provided)"
                    },
                    "root_concept": {
                        "type": "string",
                        "description": "Main concept of the document"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Knowledge domain",
                        "enum": ["ai_architecture", "prompt_engineering", "neural_networks", "cognitive_science", "software_engineering", "research_methods", "machine_learning", "natural_language_processing", "general_knowledge"]
                    },
                    "use_fetch_enhanced_context": {
                        "type": "boolean",
                        "description": "Whether to use fetch-enhanced context for processing (leverages real-time data)",
                        "default": False
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata for the document",
                        "properties": {
                            "source": {"type": "string"},
                            "version": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "required": ["root_concept", "domain"],
                "anyOf": [
                    {"required": ["document_content"]},
                    {"required": ["source_url"]}
                ]
            },
        ),
        types.Tool(
            name="get_concept",
            description="Retrieve specific concept from document knowledge",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_name": {
                        "type": "string",
                        "description": "Name of concept to retrieve"
                    }
                },
                "required": ["concept_name"]
            },
        ),
        types.Tool(
            name="browse_knowledge_shelf",
            description="Browse concepts in a knowledge domain",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Knowledge domain to browse",
                        "enum": ["ai_architecture", "prompt_engineering", "neural_networks", "cognitive_science", "software_engineering", "research_methods", "machine_learning", "natural_language_processing", "general_knowledge"]
                    }
                },
                "required": ["domain"]
            },
        ),
        types.Tool(
            name="query_blended_knowledge",
            description="Query both conversation and document knowledge simultaneously",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query"
                    },
                    "include_formal": {
                        "type": "boolean",
                        "description": "Include formal document knowledge",
                        "default": True
                    },
                    "include_conversational": {
                        "type": "boolean",
                        "description": "Include conversation insights",
                        "default": True
                    }
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="link_conversation_to_knowledge",
            description="Create cross-references between conversation and document knowledge",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "ID of conversation to analyze for concept mentions"
                    },
                    "document_concept_id": {
                        "type": "string",
                        "description": "Optional specific concept to link to"
                    }
                },
                "required": ["conversation_id"]
            },
        ),
        types.Tool(
            name="get_available_models",
            description="Get a list of available models from the provider.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            },
        ),
        types.Tool(
            name="get_current_model",
            description="Get the currently selected model.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            },
        ),
        types.Tool(
            name="set_model",
            description="Set the model to use for generation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The name of the model to set."
                    }
                },
                "required": ["model_name"]
            },
        ),
        # Enhanced Knowledge Tools
        types.Tool(
            name="store_knowledge_from_url",
            description="Store knowledge by fetching content directly from a URL with intelligent merging",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch content from"},
                    "root_concept": {"type": "string", "description": "Main concept name"},
                    "domain": {
                        "type": "string",
                        "description": "Knowledge domain",
                        "enum": ["ai_architecture", "prompt_engineering", "neural_networks", "cognitive_science", "software_engineering", "research_methods", "machine_learning", "natural_language_processing", "general_knowledge"]
                    },
                    "merge_with_existing": {"type": "boolean", "description": "Whether to merge with existing concepts", "default": True},
                    "metadata": {"type": "object", "description": "Additional metadata"}
                },
                "required": ["url", "root_concept", "domain"]
            },
        ),
        types.Tool(
            name="store_knowledge_from_search",
            description="Store knowledge by searching and aggregating results with intelligent merging",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "Search query to execute"},
                    "root_concept": {"type": "string", "description": "Main concept name"},
                    "domain": {
                        "type": "string",
                        "description": "Knowledge domain",
                        "enum": ["ai_architecture", "prompt_engineering", "neural_networks", "cognitive_science", "software_engineering", "research_methods", "machine_learning", "natural_language_processing", "general_knowledge"]
                    },
                    "max_results": {"type": "integer", "description": "Maximum number of search results", "default": 3, "minimum": 1, "maximum": 10},
                    "merge_with_existing": {"type": "boolean", "description": "Whether to merge with existing concepts", "default": True},
                    "metadata": {"type": "object", "description": "Additional metadata"}
                },
                "required": ["search_query", "root_concept", "domain"]
            },
        ),
        types.Tool(
            name="enhance_existing_concept",
            description="Enhance an existing concept with additional information from multiple sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_name": {"type": "string", "description": "Name of existing concept to enhance"},
                    "domain": {
                        "type": "string",
                        "description": "Knowledge domain",
                        "enum": ["ai_architecture", "prompt_engineering", "neural_networks", "cognitive_science", "software_engineering", "research_methods", "machine_learning", "natural_language_processing", "general_knowledge"]
                    },
                    "enhancement_sources": {
                        "type": "array",
                        "description": "List of URLs, search queries, or file paths",
                        "items": {"type": "string"}
                    },
                    "source_type": {"type": "string", "description": "Type of sources", "enum": ["url", "search", "file", "auto"], "default": "auto"}
                },
                "required": ["concept_name", "domain", "enhancement_sources"]
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool execution."""
    try:
        cme = await initialize_engine()

        if name == "store_conversation":
            conversation = arguments["conversation"]
            context = arguments.get("context", {})

            try:
                # Create a task for background processing
                task_id = task_manager.create_task(
                    description=f"Storing conversation with {len(conversation)} messages"
                )

                # Start background processing
                await task_manager.start_background_task(
                    task_id,
                    cme.store_conversation,
                    conversation,
                    context
                )

                # Return immediately with task info
                response = create_immediate_response(
                    task_id,
                    f"Processing conversation with {len(conversation)} messages"
                )

                return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

            except Exception as e:
                logger.error(f"Error starting conversation storage: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to start conversation storage: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "query_memory":
            query = arguments["query"]
            context_depth = arguments.get("context_depth", 3)
            time_scope = arguments.get("time_scope", "week")
            max_results = arguments.get("max_results", 10)

            try:
                results = await cme.query_memory(
                    query=query,
                    context_depth=context_depth,
                    time_scope=time_scope,
                    max_results=max_results
                )
                response_text = json.dumps(results, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in query_memory: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to query memory: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "generate_response":
            prompt = arguments["prompt"]
            context_depth = arguments.get("context_depth", 3)
            response_style = arguments.get("response_style", "conversational")

            try:
                response = await cme.generate_response(
                    prompt=prompt,
                    context_depth=context_depth,
                    response_style=response_style
                )
                return [types.TextContent(type="text", text=response)]
            except Exception as e:
                logger.error(f"Error in generate_response: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to generate response: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "analyze_conversation":
            conversation_id = arguments.get("conversation_id")
            analysis_type = arguments.get("analysis_type", "all")

            try:
                analysis = await cme.analyze_conversation(
                    conversation_id=conversation_id,
                    analysis_type=analysis_type
                )
                response_text = json.dumps(analysis, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in analyze_conversation: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to analyze conversation: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "get_memory_stats":
            include_details = arguments.get("include_details", False)

            try:
                stats = await cme.get_memory_stats(include_details=include_details)
                response_text = json.dumps(stats, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in get_memory_stats: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to get memory stats: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "get_rtm_tree_details":
            tree_id = arguments["tree_id"]
            try:
                if cme.rtm_store:
                    details = await cme.rtm_store.get_tree_statistics(tree_id)
                    response_text = json.dumps(details, indent=2, default=str)
                    return [types.TextContent(type="text", text=response_text)]
                else:
                    raise CMEError("RTM store not initialized.")
            except Exception as e:
                logger.error(f"Error in get_rtm_tree_details: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to get RTM tree details: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "get_vector_store_details":
            include_details = arguments.get("include_details", False)
            try:
                stats = await cme.get_memory_stats(include_details=include_details)
                vector_store_details = stats.get("vector_store", {})
                response_text = json.dumps(vector_store_details, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in get_vector_store_details: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to get vector store details: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "get_task_status":
            task_id = arguments["task_id"]
            try:
                task_info = task_manager.get_task_status(task_id)
                if task_info:
                    response = task_info.to_dict()
                else:
                    response = {
                        "status": "error",
                        "error": f"Task {task_id} not found",
                        "timestamp": datetime.now().isoformat()
                    }
                return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
            except Exception as e:
                logger.error(f"Error in get_task_status: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to get task status: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "list_tasks":
            status_filter = arguments.get("status_filter", "all")
            try:
                all_tasks = task_manager.get_all_tasks()

                if status_filter != "all":
                    # Filter by status
                    filtered_tasks = {
                        task_id: task_info
                        for task_id, task_info in all_tasks.items()
                        if task_info.status.value == status_filter
                    }
                else:
                    filtered_tasks = all_tasks

                # Convert to serializable format
                response = {
                    "total_tasks": len(filtered_tasks),
                    "status_filter": status_filter,
                    "tasks": [task_info.to_dict() for task_info in filtered_tasks.values()],
                    "timestamp": datetime.now().isoformat()
                }

                return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
            except Exception as e:
                logger.error(f"Error in list_tasks: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to list tasks: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "get_available_models":
            try:
                models = await cme.get_available_models()
                response_text = json.dumps(models, indent=2)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in get_available_models: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to get available models: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "get_current_model":
            try:
                model = cme.get_current_model()
                response_text = json.dumps({"current_model": model}, indent=2)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in get_current_model: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to get current model: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "store_document_knowledge":
            document_content = arguments.get("document_content")
            source_url = arguments.get("source_url")
            root_concept = arguments["root_concept"]
            domain = arguments["domain"]
            use_fetch_enhanced_context = arguments.get("use_fetch_enhanced_context", False)
            metadata = arguments.get("metadata", {})

            try:
                # If source_url is provided, fetch content
                if source_url and not document_content:
                    logger.info(f"Fetching content from URL: {source_url}")
                    if enhanced_tools:
                        try:
                            document_content = await enhanced_tools._fetch_content_from_url(source_url)
                            # Add source URL to metadata
                            metadata["source"] = source_url
                            metadata["fetched_at"] = datetime.now().isoformat()
                        except Exception as fetch_error:
                            logger.warning(f"Failed to fetch from URL, proceeding with error: {fetch_error}")
                            error_response = {
                                "status": "error",
                                "error": f"Failed to fetch content from URL: {str(fetch_error)}",
                                "url": source_url,
                                "timestamp": datetime.now().isoformat()
                            }
                            return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]
                    else:
                        error_response = {
                            "status": "error",
                            "error": "Enhanced tools not available for URL fetching",
                            "timestamp": datetime.now().isoformat()
                        }
                        return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

                # Ensure we have content to process
                if not document_content:
                    error_response = {
                        "status": "error",
                        "error": "Either document_content or source_url must be provided",
                        "timestamp": datetime.now().isoformat()
                    }
                    return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

                # If fetch-enhanced context is requested, pre-process through context assembler
                if use_fetch_enhanced_context and cme.context_assembler and cme.context_assembler.enhanced_knowledge_tools:
                    logger.info("Using fetch-enhanced context for document processing")
                    try:
                        # Use context assembler to enrich the content with related fresh data
                        enhanced_context = await cme.context_assembler.assemble_context(
                            query=f"Related information for {root_concept} in {domain}",
                            strategy="fetch_enhanced",
                            max_depth=2  # Limit traversal depth for processing
                        )

                        # Add context enrichment note to metadata
                        metadata["context_enhanced"] = True
                        metadata["enhancement_nodes"] = len(enhanced_context.retrieved_nodes)

                        # The document content remains the primary content, but context is enriched
                        logger.info(f"Enhanced context with {len(enhanced_context.retrieved_nodes)} additional nodes")

                    except Exception as context_error:
                        logger.warning(f"Context enhancement failed, proceeding without: {context_error}")
                        metadata["context_enhancement_failed"] = str(context_error)

                # Store the document knowledge
                result = await cme.store_document_knowledge(
                    document_content=document_content,
                    root_concept=root_concept,
                    domain=domain,
                    metadata=metadata
                )

                # Add integration info to response
                result["integration_features"] = {
                    "fetch_enabled": source_url is not None,
                    "context_enhanced": use_fetch_enhanced_context,
                    "enhanced_tools_available": cme.context_assembler.enhanced_knowledge_tools is not None if cme.context_assembler else False
                }

                response_text = json.dumps(result, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]

            except Exception as e:
                logger.error(f"Error in store_document_knowledge: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to store document knowledge: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "get_concept":
            concept_name = arguments["concept_name"]

            try:
                concept = await cme.get_concept(concept_name)
                if concept:
                    response_text = json.dumps(concept, indent=2, default=str)
                else:
                    response = {
                        "status": "not_found",
                        "concept_name": concept_name,
                        "message": f"Concept '{concept_name}' not found in document knowledge",
                        "timestamp": datetime.now().isoformat()
                    }
                    response_text = json.dumps(response, indent=2)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in get_concept: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to retrieve concept: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "browse_knowledge_shelf":
            domain = arguments["domain"]

            try:
                shelf_data = await cme.browse_knowledge_shelf(domain)
                response_text = json.dumps(shelf_data, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in browse_knowledge_shelf: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to browse knowledge shelf: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "query_blended_knowledge":
            query = arguments["query"]
            include_formal = arguments.get("include_formal", True)
            include_conversational = arguments.get("include_conversational", True)

            try:
                result = await cme.query_blended_knowledge(
                    query=query,
                    include_formal=include_formal,
                    include_conversational=include_conversational
                )
                response_text = json.dumps(result, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in query_blended_knowledge: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to query blended knowledge: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "link_conversation_to_knowledge":
            conversation_id = arguments["conversation_id"]
            document_concept_id = arguments.get("document_concept_id")

            try:
                links = await cme.link_conversation_to_knowledge(
                    conversation_id=conversation_id,
                    document_concept_id=document_concept_id
                )
                response = {
                    "status": "success",
                    "conversation_id": conversation_id,
                    "links_created": len(links),
                    "links": links,
                    "timestamp": datetime.now().isoformat()
                }
                response_text = json.dumps(response, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in link_conversation_to_knowledge: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to link conversation to knowledge: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "set_model":
            model_name = arguments["model_name"]
            try:
                cme.set_model(model_name)
                response_text = json.dumps({"status": "success", "current_model": model_name}, indent=2)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in set_model: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to set model: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "store_knowledge_from_url":
            if not enhanced_tools:
                raise CMEError("Enhanced knowledge tools are not available.")
            try:
                result = await enhanced_tools.store_knowledge_from_url(**arguments)
                response_text = json.dumps(result, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in store_knowledge_from_url: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to store knowledge from URL: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "store_knowledge_from_search":
            if not enhanced_tools:
                raise CMEError("Enhanced knowledge tools are not available.")
            try:
                result = await enhanced_tools.store_knowledge_from_search(**arguments)
                response_text = json.dumps(result, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in store_knowledge_from_search: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to store knowledge from search: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        elif name == "enhance_existing_concept":
            if not enhanced_tools:
                raise CMEError("Enhanced knowledge tools are not available.")
            try:
                result = await enhanced_tools.enhance_existing_concept(**arguments)
                response_text = json.dumps(result, indent=2, default=str)
                return [types.TextContent(type="text", text=response_text)]
            except Exception as e:
                logger.error(f"Error in enhance_existing_concept: {str(e)}")
                error_response = {
                    "status": "error",
                    "error": f"Failed to enhance existing concept: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        error_response = {
            "error": str(e),
            "tool": name,
            "arguments": arguments,
            "type": "tool_execution_error"
        }
        return [types.TextContent(type="text", text=json.dumps(error_response, indent=2))]


async def main() -> int:
    """Main server execution function."""
    exit_code = 0
    try:
        logger.info("Starting Cognitive Memory Engine MCP server")

        # Set up stdio transport
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            try:
                await server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="cognitive-memory-engine",
                        server_version="0.1.0",
                        capabilities=server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
            except Exception as server_error:
                logger.error(f"Server execution error: {server_error}")
                # Don't re-raise here to ensure cleanup happens
                exit_code = 1

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        exit_code = 0
    except Exception as e:
        logger.error(f"Server error: {e}")
        exit_code = 1
    finally:
        # Proper async cleanup - ensure this always runs
        try:
            await async_cleanup()
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")
            if exit_code == 0:  # Only change exit code if no previous error
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Server interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
