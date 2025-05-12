# Fix collection name resolution and document serialization in DeepWiki chat API

This commit resolves two critical issues in the DeepWiki chat API:

1. **Document Serialization Error**: Fixed the 500 Internal Server Error that occurred when accessing certain repositories like "customs_exchange_rate_main". LangChain Document objects in the retrieved_documents field couldn't be serialized to JSON. Added proper serialization logic in:
   - api/langgraph/graph.py - Added document serialization in run_retrieval_only_pipeline and debug_rag_pipeline
   - api/langgraph/chat.py - Added document serialization in get_chat_response

2. **Collection Name Resolution**: Moved collection name resolution logic from frontend to backend for better maintainability:
   - Enhanced resolve_repo_path_for_chat to try multiple variations of repo IDs
   - Added automatic variation matching in get_chat_response
   - Removed hardcoded collection mappings from frontend
   - Implemented robust error handling for collection lookup

3. **Field Name Mismatch Fix**: Fixed the mismatch between "retrieved_documents" in retrieve_node and "relevant_documents" in generate_node to ensure retrieved documents are properly passed to the generation step.

4. **API Improvements**:
   - Updated API to work with collection names directly without requiring valid file paths
   - Added better logging for collection resolution
   - Made collection lookup completely transparent to clients

These changes make the chat functionality more robust and eliminate the need for frontend changes when adding new repositories with unique naming patterns.

Testing:
- Added test_document_serialization.py to verify serialization logic
- Added test_collection_resolution.py to verify backend-only collection resolution
- Updated test_chat_customs.py to work with the new implementation

Documentation:
- Updated FIXED_ISSUES.md with detailed explanation of the fixes
- Updated TASKS.md to mark tasks as completed 