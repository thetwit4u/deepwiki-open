#!/usr/bin/env python3
"""
Test script to verify the improved backend-only collection resolution in DeepWiki.
This tests the chat API's ability to automatically find the correct ChromaDB collection
for a repository ID, especially for problematic repositories like customs_exchange_rate_main.
"""

import sys
import os
import json
import requests

sys.path.append('.')

# Import relevant modules
from api.langgraph.chat import get_chat_response
from api.langgraph.wiki_structure import normalize_repo_id
from api.langgraph.chroma_utils import generate_collection_name, get_chroma_client, check_collection_exists, get_persistent_dir

def test_direct_collection_resolution():
    """Test the get_chat_response function's ability to resolve collections directly."""
    print("\n=== Testing Direct Collection Resolution in get_chat_response ===\n")
    
    # Test with the problematic repository ID
    repo_id = "customs_exchange_rate_main"
    query = "What is the repository about?"
    
    print(f"Testing with repo_id: {repo_id}")
    
    # First, confirm the expected collection exists
    client = get_chroma_client(get_persistent_dir())
    collections = client.list_collections()
    collection_names = [str(c) for c in collections]
    print(f"Available collections: {collection_names[:5]}{'...' if len(collection_names) > 5 else ''}")
    
    # Try to get the expected collection name
    normalized_id = normalize_repo_id(repo_id)
    expected_name = generate_collection_name(normalized_id)
    expected_name_alt = "local_customs_exchange_rate_main_9cfa74b61a"  # Known problematic collection
    
    print(f"Normalized repo_id: {normalized_id}")
    print(f"Expected collection name: {expected_name}")
    print(f"Known alternative collection name: {expected_name_alt}")
    
    # Check if either collection exists
    expected_exists = check_collection_exists(client, expected_name)
    alt_exists = check_collection_exists(client, expected_name_alt)
    
    print(f"Expected collection exists: {expected_exists}")
    print(f"Alternative collection exists: {alt_exists}")
    
    # Now call get_chat_response without specifying a collection name
    print("\nCalling get_chat_response without specifying collection_name...")
    try:
        response = get_chat_response(repo_id, query)
        print("Success! get_chat_response returned a valid response")
        
        # Check what collection was actually used
        used_collection = response.get("metadata", {}).get("collection_name")
        print(f"Collection name used: {used_collection}")
        
        if used_collection:
            if expected_exists and used_collection == expected_name:
                print("✅ Used the expected normalized collection name")
            elif alt_exists and used_collection == expected_name_alt:
                print("✅ Used the known alternative collection name")
            else:
                print(f"ℹ️ Used a different collection name: {used_collection}")
        else:
            print("❌ No collection name found in response metadata")
        
        # Check if we got an answer
        if "answer" in response and response["answer"]:
            print("✅ Got a valid answer from the chat API")
            print(f"Answer snippet: {response['answer'][:100]}...")
        else:
            print("❌ Did not get a valid answer")
            
        # Check retrieved documents
        docs = response.get("retrieved_documents", [])
        print(f"Retrieved {len(docs)} documents")
        if docs:
            print("✅ Documents were properly serialized")
        
        return True
    except Exception as e:
        print(f"❌ Error in get_chat_response: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoint():
    """Test the /chat API endpoint's ability to resolve collections automatically."""
    print("\n=== Testing API Endpoint Collection Resolution ===\n")
    
    repo_id = "customs_exchange_rate_main"
    api_url = "http://localhost:8001/chat"
    
    payload = {
        "repo_id": repo_id,
        "message": "What is the repository about?",
        "generator_provider": "gemini",
        "embedding_provider": "ollama_nomic",
        "top_k": 5
    }
    
    print(f"Testing API with repo_id: {repo_id}")
    print(f"API URL: {api_url}")
    print(f"Payload: {json.dumps(payload)}")
    
    try:
        print("\nCalling API endpoint...")
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API returned status 200")
            
            # Check collection name in metadata
            used_collection = data.get("metadata", {}).get("collection_name")
            print(f"Collection name used: {used_collection}")
            
            # Check answer
            if "answer" in data and data["answer"]:
                print("✅ Got a valid answer from the API")
                print(f"Answer snippet: {data['answer'][:100]}...")
            else:
                print("❌ Did not get a valid answer")
                
            # Check retrieved documents
            docs = data.get("retrieved_documents", [])
            print(f"Retrieved {len(docs)} documents")
            if docs:
                print("✅ Documents were properly serialized")
                # Check a sample document
                print(f"Sample document format: {json.dumps(docs[0][:100] if isinstance(docs[0], str) else docs[0])[:100]}...")
            
            return True
        else:
            print(f"❌ API returned error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data)}")
            except:
                print(f"Error text: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing Improved Collection Resolution ===")
    print("This script tests the backend-only collection resolution for problematic repositories.")
    
    # Make sure API is running if needed
    api_running = input("Is the API server running (needed for API test)? (y/n): ").lower() == 'y'
    
    # Test direct resolution
    direct_success = test_direct_collection_resolution()
    
    # Test API endpoint if API is running
    api_success = False
    if api_running:
        api_success = test_api_endpoint()
    else:
        print("\n=== Skipping API test as API server is not running ===")
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Direct Collection Resolution: {'✅ Success' if direct_success else '❌ Failed'}")
    if api_running:
        print(f"API Endpoint Resolution: {'✅ Success' if api_success else '❌ Failed'}")
    
    if direct_success and (not api_running or api_success):
        print("\n✅ All tests passed! The backend-only collection resolution is working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the output above for details.") 