# DeepWiki Troubleshooting Guide

## Collection Name Resolution Issues

Some repositories, particularly those with special characters in their names, may encounter collection name resolution issues when accessing via the chat API. This document explains the issue and provides solutions.

### Symptoms

You may encounter errors like:

```
Error resolving repo path: No existing collection found for [repository_id] with ollama_nomic embeddings.
```

Or you might see a 500 Internal Server Error when trying to access the chat API.

### Cause

DeepWiki uses ChromaDB to store embeddings, and collection names are generated based on the repository ID and path. When the repository has special characters or is accessed from different paths, the collection name resolution may fail.

### Solutions

#### 1. Use Direct Collection Name Parameter

The most reliable solution is to explicitly pass the collection name:

```javascript
fetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    repoId: 'repository_id',
    message: 'Your question?',
    collectionName: 'exact_collection_name'
  })
})
```

Or use the URL parameter:

```
/?repo=repository_id&collection=exact_collection_name
```

#### 2. Use the Hardcoded Workaround

For known problematic repositories, we've added a hardcoded mapping in the frontend API. Currently supported:

- `customs_exchange_rate_main` → `local_customs_exchange_rate_main_9cfa74b61a`

You can add more mappings in `frontend/src/app/api/chat/route.ts`.

### Diagnostic Tools

We've created several utilities to help diagnose and fix collection name issues:

#### `check_collections.py`

Lists all collections and finds matching ones for a specific repository:

```bash
./check_collections.py repository_id
```

#### `test_chat_customs.py`

Tests the backend API directly with collection name override:

```bash
./test_chat_customs.py
```

#### `test_frontend_workaround.py`

Tests the frontend API with the hardcoded collection workaround:

```bash
./test_frontend_workaround.py
```

#### `curl_test_customs_chat.sh`

Tests the backend API via curl with explicit collection name:

```bash
./curl_test_customs_chat.sh
```

#### Browser Test Page

A simple HTML page for testing the chat API from the browser:

```
/test_frontend_chat.html
```

### For Developers

When adding new repositories that might cause collection name resolution issues:

1. Run `./check_collections.py repository_id` to find the correct collection name
2. Add the mapping to `HARDCODED_COLLECTIONS` in `frontend/src/app/api/chat/route.ts`
3. Test the workaround with `./test_frontend_workaround.py`

For more details, see the full [Chat API Guide](./CHAT_API_GUIDE.md). 