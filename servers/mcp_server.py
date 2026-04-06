#!/usr/bin/env python3
"""
Qdrant MCP Server for Financial Documents

Exposes Qdrant document search tools to Claude Desktop.
Returns raw JSON data - Claude handles all formatting.
"""

import asyncio
import logging
import json
import yaml
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import Qdrant tools
from src.tools.qdrant_tools import (
    search_documents_semantic,
    search_by_metadata,
    compare_quarters,
    get_available_metadata,
    get_qdrant_collection_stats,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("qdrant-financial-docs")

# Load tool definitions from YAML
TOOLS_FILE = Path(__file__).parent / "tool_definitions.yaml"
with open(TOOLS_FILE) as f:
    TOOL_DEFINITIONS = yaml.safe_load(f)["tools"]

# Tool routing map
TOOL_HANDLERS = {
    "search_documents": search_documents_semantic,
    "filter_by_metadata": search_by_metadata,
    "compare_quarters": compare_quarters,
    "get_available_data": get_available_metadata,
    "get_collection_stats": get_qdrant_collection_stats,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Load tool definitions from YAML and return as MCP Tools"""
    return [
        Tool(
            name=tool["name"],
            description=tool["description"],
            inputSchema=tool["schema"]
        )
        for tool in TOOL_DEFINITIONS
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute tool and return raw JSON - Claude handles formatting"""
    
    try:
        # Check if tool exists
        if name not in TOOL_HANDLERS:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
        
        # Log the call
        logger.info(f"Tool: {name} | Args: {arguments}")
        
        # Execute the tool
        handler = TOOL_HANDLERS[name]
        result = handler.invoke(arguments)
        
        # Return raw JSON - let Claude format it beautifully
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e), "tool": name})
        )]


# ============= MAIN =============

async def main():
    """Run the MCP server"""
    logger.info("Starting Qdrant Financial Documents MCP Server")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
