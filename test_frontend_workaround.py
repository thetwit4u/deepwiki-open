#!/usr/bin/env python3
"""
Test script to verify the frontend API hardcoded collection name workaround.
"""

import requests
import json
import sys
import traceback

# Configuration
API_URL = "http://localhost:3000/api/chat"
REPO_ID = "customs_exchange_rate_main"
QUERY = "What is this repository about and what are its key components?"

def test_frontend_workaround():
    """Test the frontend API with the hardcoded collection name workaround."""
    print("Testing frontend API with hardcoded collection workaround...")
    print(f"API URL: {API_URL}")
    print(f"Repository ID: {REPO_ID}")
    print(f"Query: {QUERY}")
    print("-" * 60)
    
    # Create the request payload - deliberately NOT including collectionName
    # to test the hardcoded workaround
    payload = {
        "repoId": REPO_ID,
        "message": QUERY,
        "generatorProvider": "gemini",
        "embeddingProvider": "ollama_nomic",
        "topK": 10
        # Omit collectionName to test the hardcoded workaround
    }
    
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Send the request
        response = requests.post(API_URL, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print("\n=== API Response ===")
            if "answer" in data:
                print(f"Answer: {data['answer'][:200]}...")
                print("\n✅ SUCCESS: API response contains answer!")
                
                # Print metadata
                if "metadata" in data:
                    print("\n=== Metadata ===")
                    metadata = data["metadata"]
                    for key, value in metadata.items():
                        if key != "retrieved_documents":
                            print(f"{key}: {value}")
            else:
                print(f"No answer in response. Response keys: {list(data.keys())}")
                if "error" in data:
                    print(f"Error: {data['error']}")
        else:
            print(f"\n❌ ERROR: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            
            # Try to parse error response
            try:
                error_data = response.json()
                print(f"Error data: {json.dumps(error_data, indent=2)}")
            except:
                print("Could not parse error response as JSON")
                
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_frontend_workaround() 