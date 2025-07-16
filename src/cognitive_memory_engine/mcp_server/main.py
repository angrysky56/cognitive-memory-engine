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
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic.networks import AnyUrl

from ..config import load_env_config
from ..core.engine import CognitiveMemoryEngine
from ..core.exceptions import CMEError
from ..task_manager import task_manager, create_immediate_response

# Configure logging to stderr for MCP servers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Global engine instance
engine: CognitiveMemoryEngine | None = None

# Create MCP server
server = Server("cognitive-memory-engine")


async def initialize_engine() -> CognitiveMemoryEngine:
    """Initialize the Cognitive Memory Engine with environment configuration."""
    global engine

    if engine is None:
        try:
            logger.info("Loading configuration from environment...")
            config = load_env_config()
            logger.info(f"Using LLM model: {config.llm_model}")
            
            logger.info("Initializing Cognitive Memory Engine...")
            engine = CognitiveMemoryEngine(config)
            await engine.initialize()
            logger.info("Cognitive Memory Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Memory Engine: {e}")
            raise CMEError(f"Engine initialization failed: {e}") from e

    return engine


def cleanup():
    """Clean up resources on shutdown."""
    global engine

    logger.info("Shutting down Cognitive Memory Engine MCP server")

    if engine:
        try:
            asyncio.create_task(engine.cleanup())
            logger.info("Engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during engine cleanup: {e}")
        finally:
            engine = None


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down gracefully")
    cleanup()
    sys.exit(0)


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
            description="Access to recent conversation history",
            mimeType="application/json",
        ),
        types.Resource(
            uri=AnyUrl("cme://memory/narratives"),
            name="Narrative Trees",
            description="Hierarchical narrative structures from conversations",
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
            return json.dumps(conversations, indent=2)

        elif uri == "cme://memory/narratives":
            narratives = await cme.get_narrative_summaries()
            return json.dumps(narratives, indent=2)

        elif uri == "cme://memory/temporal":
            temporal_data = await cme.get_temporal_organization()
            return json.dumps(temporal_data, indent=2)

        elif uri == "cme://memory/context":
            context = await cme.get_active_context()
            return json.dumps(context, indent=2)

        elif uri == "cme://memory/vector_store":
            stats = await cme.get_memory_stats(include_details=True)
            vector_store_details = stats.get("vector_store", {})
            return json.dumps(vector_store_details, indent=2)

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
                models = await cme.llm_provider.get_available_models()
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
                model = cme.llm_provider.get_current_model()
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

        elif name == "set_model":
            model_name = arguments["model_name"]
            try:
                cme.llm_provider.set_model(model_name)
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

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        exit_code = 0
    except Exception as e:
        logger.error(f"Server error: {e}")
        exit_code = 1
    finally:
        cleanup()
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
