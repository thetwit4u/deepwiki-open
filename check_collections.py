#!/usr/bin/env python3
"""
Utility script to check ChromaDB collections and map them to repositories.
"""

import sys
sys.path.append('.')

from api.langgraph.chroma_utils import get_chroma_client, get_persistent_dir
from api.langgraph.wiki_structure import normalize_repo_id
import re

def list_collections(repo_filter=None):
    """
    List all collections in ChromaDB, optionally filtering by repository ID.
    
    Args:
        repo_filter: Optional repository ID to filter by
    """
    # Get ChromaDB client
    persistent_dir = get_persistent_dir()
    print(f"ChromaDB directory: {persistent_dir}")
    
    client = get_chroma_client(persistent_dir)
    collections = client.list_collections()
    
    print(f"Total collections: {len(collections)}")
    
    if repo_filter:
        normalized_filter = normalize_repo_id(repo_filter)
        print(f"\nFiltering collections for repository: {repo_filter}")
        print(f"Normalized repository ID: {normalized_filter}")
        
        # Find collections that match the repository ID
        # In ChromaDB v0.6.0+, collection names are returned directly as strings
        matching_collections = [c for c in collections if normalized_filter in c]
        
        if matching_collections:
            print(f"\nFound {len(matching_collections)} matching collections:")
            for i, name in enumerate(matching_collections, 1):
                print(f"{i}. {name}")
                
            print("\n==== HOW TO USE ====")
            print("To use the collection directly in your chat requests, add the following parameter:")
            print(f"collection_name: \"{matching_collections[0]}\"")
            
            # Show example curl command
            print("\nExample API call:")
            print(f"curl -X POST http://localhost:8001/chat -H \"Content-Type: application/json\" \\\n  -d '{{\"repo_id\": \"{repo_filter}\", \"message\": \"What is this repository about?\", \"collection_name\": \"{matching_collections[0]}\"}}'")
            
            # Show example frontend code
            print("\nFrontend usage:")
            print(f"""
// In your frontend code
const response = await fetch('/api/chat', {{
  method: 'POST',
  headers: {{ 'Content-Type': 'application/json' }},
  body: JSON.stringify({{
    repoId: '{repo_filter}',
    message: 'What is this repository about?',
    collectionName: '{matching_collections[0]}'
  }})
}});
""")

            # Show example URL
            print("\nURL parameter:")
            print(f"/?repo={repo_filter}&collection={matching_collections[0]}")
        else:
            print(f"No collections found matching repository: {repo_filter}")
    else:
        # List all collections
        print("\nAll collections:")
        for i, collection in enumerate(collections, 1):
            print(f"{i}. {collection}")
            
        print("\nTo filter for a specific repository, run:")
        print("python check_collections.py <repository_id>")

if __name__ == "__main__":
    repo_filter = sys.argv[1] if len(sys.argv) > 1 else None
    list_collections(repo_filter) 