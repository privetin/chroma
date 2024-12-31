import asyncio
import os
import time
import logging
from typing import Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from chromadb.api.types import Document, Documents

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio
import functools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
BACKOFF_FACTOR = 2

class ErrorType:
    """Standard error types for consistent messaging"""
    NOT_FOUND = "Not found"
    ALREADY_EXISTS = "Already exists" 
    INVALID_INPUT = "Invalid input"
    FILTER_ERROR = "Filter error"
    OPERATION_ERROR = "Operation failed"

class DocumentOperationError(Exception):
    """Custom error for document operations"""
    def __init__(self, error: str):
        self.error = error
        super().__init__(self.error)

def retry_operation(operation_name: str):
    """Decorator to retry document operations with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except DocumentOperationError as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Clean up error message
                        msg = str(e)
                        if msg.lower().startswith(operation_name.lower()):
                            msg = msg[len(operation_name):].lstrip(': ')
                        if msg.lower().startswith('failed'):
                            msg = msg[7:].lstrip(': ')
                        if msg.lower().startswith('search failed'):
                            msg = msg[13:].lstrip(': ')
                        
                        # Map error patterns to friendly messages
                        error_msg = msg.lower()
                        doc_id = kwargs.get('arguments', {}).get('document_id')
                        
                        if "not found" in error_msg:
                            error = f"Document not found{f' [id={doc_id}]' if doc_id else ''}"
                        elif "already exists" in error_msg:
                            error = f"Document already exists{f' [id={doc_id}]' if doc_id else ''}"
                        elif "invalid" in error_msg:
                            error = "Invalid input"
                        elif "filter" in error_msg:
                            error = "Invalid filter"
                        else:
                            error = "Operation failed"
                            
                        raise DocumentOperationError(error)
                    await asyncio.sleep(2 ** attempt)
            return None
        return wrapper
    return decorator

# Initialize Chroma client with persistence
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

client = chromadb.Client(Settings(
    persist_directory=data_dir,
    is_persistent=True
))

try:
    collection = client.get_collection("documents")
    logger.info("Retrieved existing collection 'documents'")
except Exception:
    collection = client.create_collection("documents")
    logger.info("Created new collection 'documents'")

# Use sentence transformers for better embeddings
model_name = "all-MiniLM-L6-v2"
logger.info(f"Initializing embedding function with model: {model_name}")

try:
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    logger.info("Embedding function initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding function: {str(e)}")
    raise

# Add a sample document if collection is empty
try:
    if collection.count() == 0:
        logger.info("Adding sample document to empty collection")
        collection.add(
            documents=[
                "Vector databases are specialized databases designed to store and retrieve high-dimensional vectors efficiently. "
                "In machine learning, they are crucial for similarity search, recommendation systems, and semantic search applications. "
                "They use techniques like LSH or HNSW for fast approximate nearest neighbor search."
            ],
            ids=["sample_doc"],
            metadatas=[{
                "topic": "vector databases",
                "type": "sample",
                "date": "2024-12-31"
            }]
        )
        logger.info("Sample document added successfully")
except Exception as e:
    logger.error(f"Error adding sample document: {e}")

server = Server("chroma")

# Server command options
server.command_options = {
    "create_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "content": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": True}
        },
        "required": ["document_id", "content"]
    },
    "read_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"}
        },
        "required": ["document_id"]
    },
    "update_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"},
            "content": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": True}
        },
        "required": ["document_id", "content"]
    },
    "delete_document": {
        "type": "object",
        "properties": {
            "document_id": {"type": "string"}
        },
        "required": ["document_id"]
    },
    "list_documents": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 10},
            "offset": {"type": "integer", "minimum": 0, "default": 0}
        }
    },
    "search_similar": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "minimum": 1, "default": 5},
            "metadata_filter": {"type": "object", "additionalProperties": True},
            "content_filter": {"type": "string"}
        },
        "required": ["query"]
    }
}

def sanitize_metadata(metadata: dict) -> dict:
    """Convert metadata values to strings for Chroma compatibility"""
    if not metadata:
        return {}
    return {k: str(v) for k, v in metadata.items()}

def build_where_clause(metadata: dict) -> dict:
    """Build a valid Chroma where clause for multiple metadata conditions"""
    if not metadata:
        return {}
    
    def process_value(value):
        """Process value based on type"""
        if isinstance(value, (int, float)):
            # Keep numeric values as strings for Chroma
            return str(value)
        return str(value)
    
    conditions = []
    for key, value in metadata.items():
        if value is None:
            continue
            
        if isinstance(value, dict) and any(k.startswith('$') for k in value.keys()):
            # Handle operator conditions
            processed_value = {}
            for op, val in value.items():
                if isinstance(val, (list, tuple)):
                    # Handle array operators like $in
                    processed_value[op] = [process_value(v) for v in val]
                else:
                    # Handle single value operators
                    processed_value[op] = process_value(val)
            conditions.append({key: processed_value})
        else:
            # Simple equality condition
            conditions.append({key: {"$eq": process_value(value)}})
    
    if not conditions:
        return {}
        
    if len(conditions) == 1:
        return conditions[0]
    
    return {"$and": conditions}

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for document operations."""
    return [
        types.Tool(
            name="create_document",
            description="Create a new document in the Chroma vector database",
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
            name="read_document",
            description="Retrieve a document from the Chroma vector database by its ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"}
                },
                "required": ["document_id"]
            }
        ),
        types.Tool(
            name="update_document",
            description="Update an existing document in the Chroma vector database",
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
            name="delete_document",
            description="Delete a document from the Chroma vector database by its ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"}
                },
                "required": ["document_id"]
            }
        ),
        types.Tool(
            name="list_documents",
            description="List all documents stored in the Chroma vector database with pagination",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "default": 10},
                    "offset": {"type": "integer", "minimum": 0, "default": 0}
                }
            }
        ),
        types.Tool(
            name="search_similar",
            description="Search for semantically similar documents in the Chroma vector database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "minimum": 1, "default": 5},
                    "metadata_filter": {"type": "object", "additionalProperties": True},
                    "content_filter": {"type": "string"}
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
        arguments = {}

    try:
        if name == "create_document":
            return await handle_create_document(arguments)
        elif name == "read_document":
            return await handle_read_document(arguments)
        elif name == "update_document":
            return await handle_update_document(arguments)
        elif name == "delete_document":
            return await handle_delete_document(arguments)
        elif name == "list_documents":
            return await handle_list_documents(arguments)
        elif name == "search_similar":
            return await handle_search_similar(arguments)
        
        raise ValueError(f"Unknown tool: {name}")

    except DocumentOperationError as e:
        return [
            types.TextContent(
                type="text",
                text=f"{e.error}"
            )
        ]
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )
        ]

@retry_operation("create_document")
async def handle_create_document(arguments: dict) -> list[types.TextContent]:
    """Handle document creation with retry logic"""
    doc_id = arguments.get("document_id")
    content = arguments.get("content")
    metadata = arguments.get("metadata")

    if not doc_id or not content:
        raise DocumentOperationError("Missing document_id or content")

    try:
        # Check if document exists using get() instead of collection.get()
        try:
            existing = collection.get(
                ids=[doc_id],
                include=['metadatas']
            )
            if existing and existing['ids']:
                raise DocumentOperationError(f"Document already exists [id={doc_id}]")
        except Exception as e:
            if "not found" not in str(e).lower():
                raise

        # Process metadata
        if metadata:
            processed_metadata = {
                k: str(v) if isinstance(v, (int, float)) else v
                for k, v in metadata.items()
            }
        else:
            processed_metadata = {}

        # Add document
        collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[processed_metadata]
        )

        return [
            types.TextContent(
                type="text",
                text=f"Created document '{doc_id}' successfully"
            )
        ]
    except DocumentOperationError:
        raise
    except Exception as e:
        raise DocumentOperationError(str(e))

@retry_operation("read_document")
async def handle_read_document(arguments: dict) -> list[types.TextContent]:
    """Handle document reading with retry logic"""
    doc_id = arguments.get("document_id")

    if not doc_id:
        raise DocumentOperationError("Missing document_id")

    logger.info(f"Reading document with ID: {doc_id}")

    try:
        result = collection.get(ids=[doc_id])
        
        if not result or not result.get('ids') or len(result['ids']) == 0:
            raise DocumentOperationError(f"Document not found [id={doc_id}]")

        logger.info(f"Successfully retrieved document: {doc_id}")
        
        # Format the response
        doc_content = result['documents'][0]
        doc_metadata = result['metadatas'][0] if result.get('metadatas') else {}
        
        response = [
            f"Document ID: {doc_id}",
            f"Content: {doc_content}",
            f"Metadata: {doc_metadata}"
        ]

        return [
            types.TextContent(
                type="text",
                text="\n".join(response)
            )
        ]

    except Exception as e:
        raise DocumentOperationError(str(e))

@retry_operation("update_document")
async def handle_update_document(arguments: dict) -> list[types.TextContent]:
    """Handle document update with retry logic"""
    doc_id = arguments.get("document_id")
    content = arguments.get("content")
    metadata = arguments.get("metadata")

    if not doc_id or not content:
        raise DocumentOperationError("Missing document_id or content")

    logger.info(f"Updating document: {doc_id}")
    
    try:
        # Check if document exists
        existing = collection.get(ids=[doc_id])
        if not existing or not existing.get('ids'):
            raise DocumentOperationError(f"Document not found [id={doc_id}]")

        # Update document
        if metadata:
            # Keep numeric values in metadata
            processed_metadata = {
                k: v if isinstance(v, (int, float)) else str(v)
                for k, v in metadata.items()
            }
            collection.update(
                ids=[doc_id],
                documents=[content],
                metadatas=[processed_metadata]
            )
        else:
            collection.update(
                ids=[doc_id],
                documents=[content]
            )
        
        logger.info(f"Successfully updated document: {doc_id}")
        return [
            types.TextContent(
                type="text",
                text=f"Updated document '{doc_id}' successfully"
            )
        ]

    except Exception as e:
        raise DocumentOperationError(str(e))

@retry_operation("delete_document")
async def handle_delete_document(arguments: dict) -> list[types.TextContent]:
    """Handle document deletion with retry logic and network interruption handling"""
    doc_id = arguments.get("document_id")

    if not doc_id:
        raise DocumentOperationError("Missing document_id")

    logger.info(f"Attempting to delete document: {doc_id}")

    # First verify the document exists to avoid network retries for non-existent documents
    try:
        logger.info(f"Verifying document existence: {doc_id}")
        existing = collection.get(ids=[doc_id])
        if not existing or not existing.get('ids') or len(existing['ids']) == 0:
            raise DocumentOperationError(f"Document not found [id={doc_id}]")
        logger.info(f"Document found, proceeding with deletion: {doc_id}")
    except Exception as e:
        if "not found" in str(e).lower():
            raise DocumentOperationError(f"Document not found [id={doc_id}]")
        raise DocumentOperationError(str(e))

    # Attempt deletion with exponential backoff
    max_attempts = MAX_RETRIES
    current_attempt = 0
    last_error = None
    delay = RETRY_DELAY

    while current_attempt < max_attempts:
        try:
            logger.info(f"Delete attempt {current_attempt + 1}/{max_attempts} for document: {doc_id}")
            collection.delete(ids=[doc_id])
            
            # Verify deletion was successful
            try:
                check = collection.get(ids=[doc_id])
                if not check or not check.get('ids') or len(check['ids']) == 0:
                    logger.info(f"Successfully deleted document: {doc_id}")
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Deleted document '{doc_id}' successfully"
                        )
                    ]
                else:
                    raise Exception("Document still exists after deletion")
            except Exception as e:
                if "not found" in str(e).lower():
                    # This is good - means deletion was successful
                    logger.info(f"Successfully deleted document: {doc_id}")
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Deleted document '{doc_id}' successfully"
                        )
                    ]
                raise

        except Exception as e:
            last_error = e
            current_attempt += 1
            if current_attempt < max_attempts:
                logger.warning(
                    f"Delete attempt {current_attempt} failed for document {doc_id}. "
                    f"Retrying in {delay} seconds. Error: {str(e)}"
                )
                await asyncio.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                logger.error(
                    f"All delete attempts failed for document {doc_id}. "
                    f"Final error: {str(e)}", 
                    exc_info=True
                )
                raise DocumentOperationError(str(e))

    # This shouldn't be reached, but just in case
    raise DocumentOperationError("Operation failed")

@retry_operation("list_documents")
async def handle_list_documents(arguments: dict) -> list[types.TextContent]:
    """Handle document listing with retry logic"""
    limit = arguments.get("limit", 10)
    offset = arguments.get("offset", 0)

    try:
        # Get all documents
        results = collection.get(
            limit=limit,
            offset=offset,
            include=['documents', 'metadatas']
        )

        if not results or not results.get('ids'):
            return [
                types.TextContent(
                    type="text",
                    text="No documents found in collection"
                )
            ]

        # Format results
        response = [f"Documents (showing {len(results['ids'])} results):"]
        for i, (doc_id, content, metadata) in enumerate(
            zip(results['ids'], results['documents'], results['metadatas'])
        ):
            response.append(f"\nID: {doc_id}")
            response.append(f"Content: {content}")
            if metadata:
                response.append(f"Metadata: {metadata}")

        return [
            types.TextContent(
                type="text",
                text="\n".join(response)
            )
        ]
    except Exception as e:
        raise DocumentOperationError(str(e))

@retry_operation("search_similar")
async def handle_search_similar(arguments: dict) -> list[types.TextContent]:
    """Handle similarity search with retry logic"""
    query = arguments.get("query")
    num_results = arguments.get("num_results", 5)
    metadata_filter = arguments.get("metadata_filter")
    content_filter = arguments.get("content_filter")

    if not query:
        raise DocumentOperationError("Missing query")

    try:
        # Build query parameters
        query_params = {
            "query_texts": [query],
            "n_results": num_results,
            "include": ['documents', 'metadatas', 'distances']
        }

        # Process metadata filter
        if metadata_filter:
            where_conditions = []
            for key, value in metadata_filter.items():
                if isinstance(value, (int, float)):
                    where_conditions.append({key: {"$eq": str(value)}})
                elif isinstance(value, dict):
                    # Handle operator conditions
                    processed_value = {}
                    for op, val in value.items():
                        if isinstance(val, (list, tuple)):
                            processed_value[op] = [str(v) if isinstance(v, (int, float)) else v for v in val]
                        else:
                            processed_value[op] = str(val) if isinstance(val, (int, float)) else val
                    where_conditions.append({key: processed_value})
                else:
                    where_conditions.append({key: {"$eq": str(value)}})
            
            if len(where_conditions) == 1:
                query_params["where"] = where_conditions[0]
            else:
                query_params["where"] = {"$and": where_conditions}

        # Add content filter
        if content_filter:
            query_params["where_document"] = {"$contains": content_filter}

        # Execute search
        logger.info(f"Executing search with params: {query_params}")
        results = collection.query(**query_params)

        if not results or not results.get('ids') or len(results['ids'][0]) == 0:
            msg = ["No documents found matching query: " + query]
            if metadata_filter:
                msg.append(f"Metadata filter: {metadata_filter}")
            if content_filter:
                msg.append(f"Content filter: {content_filter}")
            return [types.TextContent(type="text", text="\n".join(msg))]

        # Format results
        response = ["Similar documents:"]
        for i, (doc_id, content, metadata, distance) in enumerate(
            zip(results['ids'][0], results['documents'][0], 
                results['metadatas'][0], results['distances'][0])
        ):
            response.append(f"\n{i+1}. Document '{doc_id}' (distance: {distance:.4f})")
            response.append(f"   Content: {content}")
            if metadata:
                response.append(f"   Metadata: {metadata}")

        return [types.TextContent(type="text", text="\n".join(response))]

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise DocumentOperationError(str(e))

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