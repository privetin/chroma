import asyncio
import chromadb
from chromadb.utils import embedding_functions

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

# Initialize Chroma client and collection
client = chromadb.Client()
embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_function
)

server = Server("chroma")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for document operations."""
    return [
        types.Tool(
            name="add_document",
            description="Add a new document with text content",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["document_id", "content"]
            }
        ),
        types.Tool(
            name="search_similar",
            description="Search for similar documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "minimum": 1, "default": 5}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    """Handle document operations."""
    if not arguments:
        raise ValueError("Missing arguments")

    if name == "add_document":
        doc_id = arguments.get("document_id")
        content = arguments.get("content")
        metadata = arguments.get("metadata", {})

        if not doc_id or not content:
            raise ValueError("Missing document_id or content")

        collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[metadata]
        )

        return [
            types.TextContent(
                type="text",
                text=f"Added document '{doc_id}' successfully"
            )
        ]

    elif name == "search_similar":
        query = arguments.get("query")
        num_results = arguments.get("num_results", 5)

        if not query:
            raise ValueError("Missing query")

        results = collection.query(
            query_texts=[query],
            n_results=num_results
        )

        response = ["Similar documents:"]
        for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            response.append(f"{i+1}. Document '{doc_id}' (similarity: {1-distance:.2f})")

        return [
            types.TextContent(
                type="text",
                text="\n".join(response)
            )
        ]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="chroma",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )