# DeepWiki API Updates

This document explains recent updates to the DeepWiki API that improve functionality and fix certain issues.

## 1. Chat API Collection Name Support

### Problem
The chat API had issues with collection name resolution, particularly with repositories that have special characters in their names. This could lead to errors like:

```
Error resolving repo path: No existing collection found for customs_exchange_rate_main with ollama_nomic embeddings. Please generate the wiki first.
```

### Solution
We've added direct collection name specification support to bypass the automatic collection name resolution:

1. **Backend API Update**: The `/chat` endpoint now accepts an optional `collection_name` parameter
2. **Frontend API Route**: Updated to pass through the `collectionName` parameter to the backend
3. **Default Embeddings**: Now defaults to `ollama_nomic` embeddings for consistency

### Usage Example

#### Frontend API Call
```typescript
// Example API request with direct collection name
const response = await fetch('/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    repoId: 'customs_exchange_rate_main',
    message: 'What is this repository about?',
    collectionName: 'local_customs_exchange_rate_main_9cfa74b61a', // Direct collection name
    generatorProvider: 'gemini',
    embeddingProvider: 'ollama_nomic'
  })
});
```

#### Finding Your Collection Name
You can use the `check_collections.py` script to determine the correct collection name:

```bash
python check_collections.py
```

## 2. Mermaid Diagram Auto-Fix Enhancements

### Improvements
We've enhanced the Mermaid diagram auto-fix functionality to provide a better user experience:

1. **Increased Fix Attempts**: Now allows up to 3 fix attempts (up from 2 previously)
2. **Error Context History**: Maintains error history between attempts for better context
3. **Targeted Fix Strategies**: Added specific error-based fix instructions
4. **Improved UI**: Enhanced error display with history and helpful tips
5. **Dynamic Temperature**: Adjusts model temperature based on retry attempt

### Technical Details

- Backend prompt engineering includes detailed error analysis
- Frontend properly tracks and displays error history
- Added helpful tips for manual fixes when auto-fix fails
- More verbose error reporting to help diagnose issues

## 3. Testing These Changes

### Testing Chat API with Collection Name

Use the included `test_frontend_chat.html` file for testing:
1. Place it in the frontend public directory
2. Access it via browser at `/test_frontend_chat.html`
3. Enter your repository ID and collection name
4. Submit a question to test the API

### Finding Collection Names

To find the correct collection name for a repository:

```bash
python check_collections.py

# Or for a specific repository:
python check_collections.py customs_exchange_rate_main
```

## 4. Implementation Notes

- The hash generation for collection names depends on the absolute path of the repository on disk
- This can cause mismatches when accessing collections by repository ID alone
- Using the direct `collection_name` parameter bypasses this issue completely
- All chat functionality now defaults to `ollama_nomic` embeddings for consistency 