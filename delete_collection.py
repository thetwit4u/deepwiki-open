#!/usr/bin/env python3
"""
Script to delete a specific ChromaDB collection.
"""

import sys
sys.path.append('.')

from api.langgraph.chroma_utils import get_chroma_client, get_persistent_dir

def delete_collection(collection_name: str):
    """
    Delete a specific collection from ChromaDB.
    
    Args:
        collection_name: Name of the collection to delete
    """
    print(f"Attempting to delete collection: '{collection_name}'")
    
    # Get ChromaDB client
    persistent_dir = get_persistent_dir()
    print(f"ChromaDB directory: {persistent_dir}")
    client = get_chroma_client(persistent_dir)
    
    # List all collections before deletion
    try:
        collections = client.list_collections()
        print(f"Collections before deletion: {len(collections)}")
        for c in collections:
            print(f"- {c}")
    except Exception as e:
        print(f"Error listing collections: {e}")
    
    # Delete the collection
    if collection_name in client.list_collections():
        try:
            client.delete_collection(collection_name)
            print(f"✅ Successfully deleted collection: '{collection_name}'")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
    else:
        print(f"❌ Collection '{collection_name}' not found")
    
    # List all collections after deletion
    try:
        collections = client.list_collections()
        print(f"\nCollections after deletion: {len(collections)}")
        for c in collections:
            print(f"- {c}")
    except Exception as e:
        print(f"Error listing collections: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
    else:
        collection_name = "local_customs_exchange-rate-main_a19b3b8e44"
        print(f"No collection name provided, using default: {collection_name}")
    
    delete_collection(collection_name) 