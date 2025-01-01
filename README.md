# Chroma MCP Server

A Model Context Protocol (MCP) server implementation that provides vector database capabilities through Chroma. This server enables semantic document search, metadata filtering, and document management with persistent storage.

## Requirements

- Python 3.8+
- Chroma 0.4.0+
- MCP SDK 0.1.0+

## Components

### Resources
The server provides document storage and retrieval through Chroma's vector database:
- Stores documents with content and metadata
- Persists data in `src/chroma/data` directory
- Supports semantic similarity search

### Tools

The server implements CRUD operations and search functionality:

#### Document Management
- `create_document`: Create a new document
  - Required: `document_id`, `content`
  - Optional: `metadata` (key-value pairs)
  - Returns: Success confirmation
  - Error: Already exists, Invalid input

- `read_document`: Retrieve a document by ID
  - Required: `document_id`
  - Returns: Document content and metadata
  - Error: Not found

- `update_document`: Update an existing document
  - Required: `document_id`, `content`
  - Optional: `metadata`
  - Returns: Success confirmation
  - Error: Not found, Invalid input

- `delete_document`: Remove a document
  - Required: `document_id`
  - Returns: Success confirmation
  - Error: Not found

- `list_documents`: List all documents
  - Optional: `limit`, `offset`
  - Returns: List of documents with content and metadata

#### Search Operations
- `search_similar`: Find semantically similar documents
  - Required: `query`
  - Optional: `num_results`, `metadata_filter`, `content_filter`
  - Returns: Ranked list of similar documents with distance scores
  - Error: Invalid filter

## Features

- **Semantic Search**: Find documents based on meaning using Chroma's embeddings
- **Metadata Filtering**: Filter search results by metadata fields
- **Content Filtering**: Additional filtering based on document content
- **Persistent Storage**: Data persists in local directory between server restarts
- **Error Handling**: Comprehensive error handling with clear messages
- **Retry Logic**: Automatic retries for transient failures

## Installation

1. Install dependencies:
```bash
uv venv
uv sync --dev --all-extras
```

## Configuration

### Claude Desktop

Add the server configuration to your Claude Desktop config:

Windows: `C:\Users\<username>\AppData\Roaming\Claude\claude_desktop_config.json`

MacOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "chroma": {
      "command": "uv",
      "args": [
        "--directory",
        "C:/MCP/server/community/chroma",
        "run",
        "chroma"
      ]
    }
  }
}
```

### Data Storage

The server stores data in:
- Windows: `src/chroma/data`
- MacOS/Linux: `src/chroma/data`

## Usage

1. Start the server:
```bash
uv run chroma
```

2. Use MCP tools to interact with the server:

```python
# Create a document
create_document({
    "document_id": "ml_paper1",
    "content": "Convolutional neural networks improve image recognition accuracy.",
    "metadata": {
        "year": 2020,
        "field": "computer vision",
        "complexity": "advanced"
    }
})

# Search similar documents
search_similar({
    "query": "machine learning models",
    "num_results": 2,
    "metadata_filter": {
        "year": 2020,
        "field": "computer vision"
    }
})
```

## Error Handling

The server provides clear error messages for common scenarios:
- `Document already exists [id=X]`
- `Document not found [id=X]`
- `Invalid input: Missing document_id or content`
- `Invalid filter`
- `Operation failed: [details]`

## Development

### Testing

1. Run the MCP Inspector for interactive testing:
```bash
npx @modelcontextprotocol/inspector uv --directory C:/MCP/server/community/chroma run chroma
```

2. Use the inspector's web interface to:
   - Test CRUD operations
   - Verify search functionality
   - Check error handling
   - Monitor server logs

### Building

1. Update dependencies:
```bash
uv compile pyproject.toml
```

2. Build package:
```bash
uv build
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
