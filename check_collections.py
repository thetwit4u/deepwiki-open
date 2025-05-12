#!/usr/bin/env python3
"""
Script to check actual collection names in ChromaDB and verify the expected collection name for the repository ID.
"""

import os
import sys
import re
sys.path.append('.')

from api.langgraph.chroma_utils import generate_collection_name, get_chroma_client, get_persistent_dir
from api.langgraph.wiki_structure import normalize_repo_id

def check_collections_for_repo(repo_id: str):
    """
    Check if a collection exists for the given repository ID and display various name transformations.
    
    Args:
        repo_id: Repository ID to check
    """
    print(f"\n===== Checking collections for '{repo_id}' =====")
    
    # Get normalized versions of the repo ID
    normalized_repo_id = normalize_repo_id(repo_id)
    print(f"Original repo ID: '{repo_id}'")
    print(f"Normalized repo ID: '{normalized_repo_id}'")
    
    # Generate all possible collection name variations
    variations = [
        repo_id,
        normalized_repo_id,
        repo_id.replace('.', '_'),
        repo_id.replace('-', '_'),
        re.sub(r'[^\w]', '_', repo_id),
        re.sub(r'[^a-zA-Z0-9]', '_', repo_id)
    ]
    
    # Remove duplicates
    variations = list(dict.fromkeys(variations))
    
    # Get ChromaDB client
    persistent_dir = get_persistent_dir()
    print(f"ChromaDB directory: {persistent_dir}")
    client = get_chroma_client(persistent_dir)
    
    # List all collections - handle ChromaDB v0.6.0
    try:
        collections = client.list_collections()
        print(f"Total collections in ChromaDB: {len(collections)}")
        
        print("\nAll collection names:")
        for c in collections:
            print(f"- {c}")
    except Exception as e:
        print(f"Error listing collections: {e}")
    
    # Check each variation
    print("\nChecking collection name variations:")
    for variation in variations:
        collection_name = generate_collection_name(variation)
        print(f"\nVariation: '{variation}'")
        print(f"Generated collection name: '{collection_name}'")
        
        # Check if this collection exists
        try:
            exists = collection_name in collections
            
            if exists:
                print(f"✅ Collection EXISTS in ChromaDB")
            else:
                print(f"❌ Collection does NOT exist in ChromaDB")
        except Exception as e:
            print(f"Error checking collection existence: {e}")

def main():
    """Main function that handles command line arguments."""
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    else:
        repo_id = "customs_exchange_rate_main"
        print(f"No repository ID provided, using default: {repo_id}")
    
    check_collections_for_repo(repo_id)

if __name__ == "__main__":
    main() 