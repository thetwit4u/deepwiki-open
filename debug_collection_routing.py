#!/usr/bin/env python3
"""
Debug script to trace the collection_name parameter processing in the chat API.
"""

import sys
sys.path.append('.')

from api.langgraph.chat import get_chat_response
import time
import json

def debug_chat_api():
    """
    Debug the chat API's handling of the collection_name parameter.
    """
    repo_id = "customs_exchange_rate_main"
    collection_name = "local_customs_exchange_rate_main_9cfa74b61a"
    query = "What is this repository about and what are its key components?"
    
    print(f"Testing chat API with direct collection name")
    print(f"Repository ID: {repo_id}")
    print(f"Collection name: {collection_name}")
    print(f"Query: {query}")
    print("-" * 60)
    
    # Patch the get_chat_response function to log parameter values
    from api.langgraph import chat as chat_module
    original_func = chat_module.get_chat_response
    
    def patched_get_chat_response(*args, **kwargs):
        print(f"TRACE: get_chat_response called with args: {args}")
        print(f"TRACE: get_chat_response called with kwargs: {kwargs}")
        return original_func(*args, **kwargs)
    
    # Apply the patch
    chat_module.get_chat_response = patched_get_chat_response
    
    # Call the API directly
    try:
        start_time = time.time()
        result = get_chat_response(
            repo_id=repo_id,
            query=query,
            generator_provider="gemini",
            embedding_provider="ollama_nomic",
            top_k=10,
            collection_name=collection_name
        )
        elapsed = time.time() - start_time
        
        print(f"\nResult received in {elapsed:.2f} seconds:")
        print("-" * 60)
        print(f"Answer: {result.get('answer', 'No answer')}")
        print("-" * 60)
        print("Metadata:")
        print(json.dumps(result.get("metadata", {}), indent=2))
        print("-" * 60)
        print(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Restore the original function
    chat_module.get_chat_response = original_func

if __name__ == "__main__":
    debug_chat_api() 