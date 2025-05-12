#!/usr/bin/env python3
"""
Script to fix the collection name issue for the customs_exchange_rate_main repository.

This script checks for the correct collection name using hashes, and provides 
a way to use the direct collection name with the chat API.
"""

import os
import sys
import re
import hashlib
sys.path.append('.')

from api.langgraph.chroma_utils import generate_collection_name, get_chroma_client, get_persistent_dir
from api.langgraph.wiki_structure import normalize_repo_id

def analyze_collection_names(repo_id: str):
    """
    Analyze the collection name issue and generate the correct one for the repository ID.
    
    Args:
        repo_id: Repository ID to analyze
    """
    print(f"\n===== Analyzing collection names for '{repo_id}' =====")
    
    # Get normalized versions of the repo ID
    normalized_repo_id = normalize_repo_id(repo_id)
    print(f"Original repo ID: '{repo_id}'")
    print(f"Normalized repo ID: '{normalized_repo_id}'")
    
    # Generate the expected collection name
    expected_collection_name = generate_collection_name(normalized_repo_id)
    print(f"Expected collection name: '{expected_collection_name}'")
    
    # Check the actual MD5 hash that should be used
    # Compute the actual hash that should be used based on absolute path in local filesystem
    repo_dir = os.path.join(os.path.expanduser('~'), 'Dev', 'ai-projects', 'deepwiki-open', 'wiki-data', 'repos', normalized_repo_id)
    abs_path = os.path.abspath(repo_dir)
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:10]
    dir_name = os.path.basename(abs_path)
    corrected_collection_name = f"local_{dir_name}_{path_hash}"
    print(f"Corrected collection name based on actual path: '{corrected_collection_name}'")
    
    # Get ChromaDB client and check existing collections
    persistent_dir = get_persistent_dir()
    print(f"ChromaDB directory: {persistent_dir}")
    client = get_chroma_client(persistent_dir)
    
    # List all collections
    try:
        collections = client.list_collections()
        print(f"Total collections in ChromaDB: {len(collections)}")
        print("\nAll collection names:")
        for c in collections:
            print(f"- {c}")
        
        # Find collections that might match our repository
        matching_collections = [c for c in collections if normalized_repo_id in c]
        print(f"\nCollections matching '{normalized_repo_id}': {len(matching_collections)}")
        for c in matching_collections:
            print(f"- {c}")
            
        if len(matching_collections) > 0:
            print(f"\n===== Solution =====")
            print(f"Use this collection name directly in your chat script:")
            print(f"COLLECTION_NAME = \"{matching_collections[0]}\"")
            print(f"\nAnd use it in your API call with the 'collection_name' parameter.")
            
            # Create a test command for convenience
            print(f"\nTest command:")
            print(f"curl -X POST http://localhost:8001/chat -H \"Content-Type: application/json\" -d '{{\"repo_id\": \"{repo_id}\", \"message\": \"What is this repository about?\", \"generator_provider\": \"gemini\", \"embedding_provider\": \"ollama_nomic\", \"collection_name\": \"{matching_collections[0]}\"}}'")
            
            return matching_collections[0]
        else:
            print(f"\nNo matching collections found. You may need to generate embeddings first.")
            return None
        
    except Exception as e:
        print(f"Error listing collections: {e}")
        return None

def main():
    """Main function that handles command line arguments."""
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    else:
        repo_id = "customs_exchange_rate_main"
        print(f"No repository ID provided, using default: {repo_id}")
    
    analyze_collection_names(repo_id)

if __name__ == "__main__":
    main() 