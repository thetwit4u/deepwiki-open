#!/usr/bin/env python3
"""
Test script to query the customs_exchange_rate_main repository via the DeepWiki chat API.
"""

import json
import requests
import sys
from pprint import pprint

# Configuration
API_URL = "http://localhost:8001/chat"
REPO_ID = "customs_exchange_rate_main"
# Use the correct collection name identified by fix_chat_customs.py script
COLLECTION_NAME = "local_customs_exchange_rate_main_9cfa74b61a"
DEFAULT_QUESTION = "Explain the main purpose of this repository and list the key components."

def query_chat(question):
    """Send a question to the DeepWiki chat API and return the response."""
    print(f"Sending query: {question}")
    print(f"Repository: {REPO_ID}")
    print(f"Using collection: {COLLECTION_NAME}")
    print("------------------------------------------------------")
    
    payload = {
        "repo_id": REPO_ID,
        "message": question,
        "generator_provider": "gemini",
        "embedding_provider": "ollama_nomic",
        "top_k": 10,
        "collection_name": COLLECTION_NAME  # Add the collection name explicitly
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)

def print_response(response):
    """Format and print the chat response."""
    print("\n=== ANSWER ===")
    print(response["answer"])
    print("\n=== RETRIEVED DOCUMENTS ===")
    if response.get("retrieved_documents"):
        for i, doc in enumerate(response["retrieved_documents"], 1):
            print(f"\n[Document {i}]")
            print(f"Source: {doc.get('source', 'N/A')}")
            if "page_content" in doc:
                content_preview = doc["page_content"][:200] + "..." if len(doc["page_content"]) > 200 else doc["page_content"]
                print(f"Content: {content_preview}")
    else:
        print("No documents retrieved")

def main():
    """Main function to get user input and query the chat API."""
    # Get the question from command line arguments or use the default
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION
    
    # Query the API
    response = query_chat(question)
    
    # Print the response
    print_response(response)
    
    # Ask if the user wants to ask another question
    while True:
        follow_up = input("\nAsk another question (leave empty to quit): ").strip()
        if not follow_up:
            break
            
        response = query_chat(follow_up)
        print_response(response)

if __name__ == "__main__":
    main() 