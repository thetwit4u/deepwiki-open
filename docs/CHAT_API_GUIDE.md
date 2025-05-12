# DeepWiki Chat API Guide

This guide explains how to use the DeepWiki Chat API, including the special collection name handling feature for repositories with resolution issues.

## Basic Chat API Usage

The DeepWiki Chat API allows you to query repository content using natural language. The basic endpoint is:

```
POST /api/chat
```

### Standard Request Format

```json
{
  "repoId": "repository_id",
  "message": "What is this repository about?",
  "generatorProvider": "gemini",
  "embeddingProvider": "ollama_nomic"
}
```

### Parameters

- `repoId` (required): The repository identifier
- `message` (required): The question or message to send
- `generatorProvider` (optional): The LLM provider to use (default: "gemini")
- `embeddingProvider` (optional): The embedding provider to use (default: "ollama_nomic")
- `topK` (optional): Number of documents to retrieve (default: 10)
- `collectionName` (optional): Direct ChromaDB collection name (for resolution issues)

## Using the Collection Name Parameter

Some repositories, especially those with special characters in their names, may encounter collection name resolution issues. For these cases, you can directly specify the ChromaDB collection name.

### Finding the Right Collection Name

Use the `check_collections.py` utility script to find the correct collection name:

```bash
python check_collections.py your_repository_id
```

### Request with Collection Name

```json
{
  "repoId": "customs_exchange_rate_main",
  "message": "What is this repository about?",
  "collectionName": "local_customs_exchange_rate_main_9cfa74b61a"
}
```

### Frontend URL Parameter

You can also specify the collection name in the URL:

```
https://your-deepwiki-instance.com/?repo=customs_exchange_rate_main&collection=local_customs_exchange_rate_main_9cfa74b61a
```

## Troubleshooting

If you encounter errors like:

```
Error resolving repo path: No existing collection found for repository_id with ollama_nomic embeddings.
```

This usually means one of two things:

1. The repository has not been indexed yet - run the wiki generation process first
2. There's a collection name resolution issue - use the `collectionName` parameter

## Testing the Chat API

You can test the Chat API using the provided utility scripts:

1. **Browser Test**: Open `/test_frontend_chat.html` in your browser
2. **Backend Test**: Run `./test_chat_customs.py` 
3. **API Test**: Run `./check_collections.py repository_id` to find the collection name

## Best Practices

1. Always use `ollama_nomic` as the embedding provider for consistency
2. For repositories with special characters in their names, use the direct collection name
3. When building applications, provide a way for users to specify the collection name 