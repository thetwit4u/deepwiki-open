#!/usr/bin/env python3
"""
Test script to query the customs_exchange_rate_main repository via the DeepWiki chat API.
"""

import sys
import os
sys.path.append('.')

from api.langgraph.chat import get_chat_response
from api.langgraph.graph import run_rag_pipeline
from api.langgraph.chroma_utils import get_chroma_client, get_persistent_dir
import traceback

def test_customs_chat():
    """Test retrieval from customs_exchange_rate_main repository."""
    
    print("ChromaDB directory:", get_persistent_dir())
    
    # Verify directory exists
    persistent_dir = get_persistent_dir()
    print("Directory exists:", os.path.exists(persistent_dir))
    
    # Get ChromaDB client
    client = get_chroma_client(persistent_dir)
    
    # List all collections
    collections = client.list_collections()
    print(f"Total collections: {len(collections)}")
    print("\nAll collections:")
    for i, collection in enumerate(collections, 1):
        # In ChromaDB v0.6.0+, collections are returned as strings
        print(f"{i}. {collection}")
    
    # Settings for the test
    repo_id = "customs_exchange_rate_main"
    query = "What is this repository about and what are its key components?"
    collection_name = None
    
    # Find collections that match our repo ID
    matching_collections = [c for c in collections if repo_id.lower() in str(c).lower()]
    print(f"\nCollections matching '{repo_id}': {len(matching_collections)}")
    
    if matching_collections:
        collection_name = matching_collections[0]
        print(f"Using collection name: {collection_name}")
        
        try:
            # Test collection access
            collection = client.get_collection(name=collection_name)
            count = collection.count()
            print(f"Collection access OK. Contains {count} embeddings.")
            
            # Prepare the query
            print(f"Query: {query}")
            
            print("\n===== Direct API Call =====")
            try:
                result = get_chat_response(
                    repo_id=repo_id,
                    query=query,
                    generator_provider="gemini",
                    embedding_provider="ollama_nomic",
                    top_k=5,
                    collection_name=collection_name
                )
                
                # Display results from direct API call
                print("\n===== Results from Direct API =====")
                print(f"Answer: {result['answer'][:200]}...")
                print(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
                print("Direct API call successful!")
                
            except Exception as e:
                print(f"\n❌ ERROR during chat test: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n===== Testing Alternate Import Method =====")
            try:
                print("Initializing RAG pipeline parameters...")
                
                # Updated call to run_rag_pipeline with the required query parameter
                result = run_rag_pipeline(
                    repo_identifier=repo_id,
                    query=query,
                    generator_provider="gemini",
                    embedding_provider="ollama_nomic",
                    collection_name=collection_name,
                    top_k=5
                )
                
                # Display results from the pipeline
                print("\n===== Results from RAG Pipeline =====")
                print(f"Answer: {result['answer'][:200]}...")
                print(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
                print("RAG pipeline successful!")
                
            except Exception as e:
                print(f"\n❌ ERROR in alternate test: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"❌ ERROR: Could not access collection: {e}")
    else:
        print(f"❌ ERROR: No collections found matching {repo_id}")
        print("Available collections:")
        for c in collections:
            print(f"  - {c}")

if __name__ == "__main__":
    test_customs_chat() 