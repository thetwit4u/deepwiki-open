#!/usr/bin/env python3
"""
Script to directly test retrieval from a ChromaDB collection without going through the chat API.
"""

import sys
from typing import Optional
from dataclasses import dataclass, field
sys.path.append('.')

from api.langgraph.chroma_utils import get_chroma_client, get_persistent_dir
from api.langgraph.embeddings import get_embedding_function
import chromadb
from langchain_community.vectorstores import Chroma

def test_direct_collection_retrieval(collection_name, query="What is this repository about?"):
    """
    Test directly retrieving documents from a ChromaDB collection.
    
    Args:
        collection_name: Name of the collection to query
        query: Query to search with
    """
    print(f"Testing direct retrieval from collection: '{collection_name}'")
    
    # Get ChromaDB client
    persistent_dir = get_persistent_dir()
    print(f"ChromaDB directory: {persistent_dir}")
    client = get_chroma_client(persistent_dir)
    
    try:
        # Check if collection exists
        collections = client.list_collections()
        print(f"Total collections: {len(collections)}")
        print("All collection names:")
        for c in collections:
            print(f"- {c}")
            
        if collection_name not in collections:
            print(f"❌ Collection '{collection_name}' does not exist!")
            return False
        
        # Create mock configuration without using dataclass's mutable defaults
        @dataclass
        class MockEmbedderConfig:
            model: str
            dimensions: Optional[int] = None
        
        # Use factory functions to avoid mutable default issue
        @dataclass
        class MockApiConfig:
            embedder: MockEmbedderConfig = field(default_factory=lambda: MockEmbedderConfig(model="text-embedding-3-small"))
            embedder_ollama: MockEmbedderConfig = field(default_factory=lambda: MockEmbedderConfig(model="nomic-embed-text"))
        
        api_config = MockApiConfig()
        
        # Get embedding function
        embedding_provider = "ollama_nomic"
        print(f"Using {embedding_provider} embeddings for retrieval")
        embedding_function = get_embedding_function(embedding_provider, api_config)
        
        # Create vector store
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        
        # Perform the search
        print(f"Searching collection with query: '{query}'")
        results = vector_store.similarity_search(query, k=3)
        
        # Print results
        print(f"Retrieved {len(results)} documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n[Document {i}]")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"Content: {content_preview}")
            
        return True
    except Exception as e:
        print(f"Error testing collection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
    else:
        collection_name = "local_customs_exchange_rate_main_9cfa74b61a"
        print(f"No collection name provided, using default: {collection_name}")
    
    test_direct_collection_retrieval(collection_name) 