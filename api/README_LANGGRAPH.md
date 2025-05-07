# DeepWiki LangGraph Implementation

This directory contains a new implementation of the DeepWiki RAG system using LangGraph/LangChain instead of adalflow.

## Overview

The LangGraph implementation provides several improvements:

1. **Modern Graph-Based Architecture**: Uses LangGraph's graph-based approach for more flexible RAG workflows.
2. **Enhanced Repository Support**: Supports both Git repositories and local directories.
3. **Improved Code Filtering**: Better exclusion patterns for common directories like `node_modules`, `.git`, etc.
4. **Persistent Vector Storage**: Uses ChromaDB for efficient, persistent vector storage.
5. **Multi-Repository Queries**: (Coming soon) Support for querying across multiple repositories.

## Files

- `langgraph_rag.py`: Core RAG pipeline implementation with LangGraph.
- `langgraph_config.py`: Configuration system for the LangGraph pipeline.
- `test_langgraph.py`: Test interface for trying the LangGraph RAG system.

## Usage

### Setup

1. Run the setup script at the root of the project:
   ```bash
   ./setup_dev.sh
   ```

2. Ensure you have API keys in your environment or `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   ```

### Testing the LangGraph RAG System

To test with a GitHub/GitLab/Bitbucket repository:
```bash
python -m api.test_langgraph --repo https://github.com/username/repository
```

To test with a local directory:
```bash
python -m api.test_langgraph --local /path/to/local/directory
```

Additional options:
- `--ollama`: Use Ollama models instead of OpenAI/Gemini (requires Ollama running locally)
- `--top-k N`: Set the number of documents to retrieve (default: 5)

## How It Works

The LangGraph implementation works in the following stages:

1. **Document Loading**: Clone repository or scan local directory, excluding irrelevant files.
2. **Text Splitting**: Split documents into chunks, using different strategies based on file type.
3. **Embedding**: Generate embeddings using OpenAI (or Ollama).
4. **Vector Storage**: Store vectors and metadata in ChromaDB with persistent storage.
5. **Retrieval**: Retrieve relevant documents from ChromaDB based on query similarity.
6. **Generation**: Generate an answer using Gemini based on the retrieved documents.
7. **Memory**: Track conversation history for context in multi-turn conversations.

## Differences from adalflow Implementation

The LangGraph implementation differs from the adalflow version in several ways:

1. **Architecture**: Uses a graph-based approach instead of adalflow's sequential pipeline.
2. **Persistence**: Uses ChromaDB for vector storage instead of pickle files.
3. **Local Path Support**: Native support for local directories in addition to Git repositories.
4. **File Filtering**: More comprehensive exclusion patterns to focus on relevant code.
5. **Configuration**: More flexible configuration system using Pydantic models.

## Project Status

This implementation is under active development. Check `TASKS.md` at the project root for details about the current status and upcoming features. 