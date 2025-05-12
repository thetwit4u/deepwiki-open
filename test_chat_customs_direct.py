#!/usr/bin/env python3
"""
Test script to directly use the get_chat_response function from our code.
"""

import sys
sys.path.append('.')

from api.langgraph.chat import get_chat_response
import json

def test_chat_direct():
    """
    Test the chat functionality by directly calling get_chat_response.
    """
    repo_id = "customs_exchange_rate_main"
    collection_name = "local_customs_exchange_rate_main_9cfa74b61a"
    query = "What is this repository about and what are its key components?"
    
    print(f"Sending query: {query}")
    print(f"Repository: {repo_id}")
    print(f"Using collection: {collection_name}")
    print("------------------------------------------------------")
    
    try:
        response = get_chat_response(
            repo_id=repo_id,
            query=query,
            generator_provider="gemini",
            embedding_provider="ollama_nomic",
            top_k=10,
            collection_name=collection_name
        )
        
        print("\n=== ANSWER ===")
        print(response.get("answer", "No answer"))
        
        print("\n=== RETRIEVED DOCUMENTS ===")
        if response.get("retrieved_documents"):
            for i, doc in enumerate(response.get("retrieved_documents"), 1):
                print(f"\n[Document {i}]")
                # Handle different document formats
                if hasattr(doc, 'metadata'):
                    # LangChain Document object
                    print(f"Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"Content: {doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content}")
                elif isinstance(doc, dict):
                    # Dictionary format
                    print(f"Source: {doc.get('source', 'N/A')}")
                    if "page_content" in doc:
                        content_preview = doc["page_content"][:200] + "..." if len(doc["page_content"]) > 200 else doc["page_content"]
                        print(f"Content: {content_preview}")
                else:
                    # Unknown format
                    print(f"Unknown document format: {type(doc)}")
                    print(f"Document: {str(doc)[:200]}...")
        else:
            print("No documents retrieved")
            
        # Write full response to file for detailed examination
        with open("chat_response_full.json", "w") as f:
            # Use a custom encoder to handle non-serializable objects
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    return str(obj)
                    
            json.dump(response, f, indent=2, cls=CustomEncoder)
            print("\nFull response written to chat_response_full.json")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def chat_interactive():
    """Interactive chat session with the model."""
    repo_id = "customs_exchange_rate_main"
    collection_name = "local_customs_exchange_rate_main_9cfa74b61a"
    
    print(f"Starting interactive chat session for repository: {repo_id}")
    print(f"Using collection: {collection_name}")
    print("Type 'exit' to quit")
    print("------------------------------------------------------")
    
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() == 'exit':
            break
            
        if not query:
            continue
            
        try:
            response = get_chat_response(
                repo_id=repo_id,
                query=query,
                generator_provider="gemini",
                embedding_provider="ollama_nomic",
                top_k=10,
                collection_name=collection_name
            )
            
            print("\n=== ANSWER ===")
            print(response.get("answer", "No answer"))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # First test with a single query
    success = test_chat_direct()
    
    if success and input("\nWould you like to start an interactive chat session? (y/n): ").lower() == 'y':
        chat_interactive() 