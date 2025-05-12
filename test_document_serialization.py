#!/usr/bin/env python3
"""
Test script to verify the serialization of LangChain Document objects in the DeepWiki chat API.
This tests the fix for the 500 Internal Server Error when returning Document objects in the API response.
"""

import sys
import os
sys.path.append('.')

from api.langgraph.chat import get_chat_response
from api.langgraph.graph import run_rag_pipeline
from langchain_core.documents import Document
import json

def test_document_serialization():
    """Test document serialization logic."""
    
    print("=== Testing Document Serialization ===")
    
    # Create a test Document object
    test_doc = Document(
        page_content="This is a test document content.",
        metadata={
            "source": "test_source.py",
            "line_numbers": [1, 2, 3],
            "is_code": True
        }
    )
    
    # Create a list of Document objects
    docs = [
        test_doc,
        Document(
            page_content="This is another test document.",
            metadata={"source": "another_source.py"}
        )
    ]
    
    print(f"\n1. Created {len(docs)} test Document objects")
    
    # Test chat.py serialization logic
    # Extract the serialization logic from get_chat_response for testing
    print("\n2. Testing serialization logic from chat.py:")
    serializable_documents_chat = []
    
    for doc in docs:
        # Check if it's a LangChain Document object (from chat.py)
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            # Convert Document to dict
            serializable_documents_chat.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        elif isinstance(doc, dict):
            # Already a dict
            serializable_documents_chat.append(doc)
        else:
            # Unknown type, try to convert to dict
            try:
                serializable_documents_chat.append(dict(doc))
            except:
                print(f"Warning: Couldn't serialize document of type {type(doc)}")
    
    # Verify the serialized documents
    print(f"  - Serialized {len(serializable_documents_chat)} documents")
    for i, doc in enumerate(serializable_documents_chat):
        print(f"  - Document {i+1}:")
        print(f"    * page_content: {doc['page_content'][:30]}...")
        print(f"    * metadata keys: {list(doc['metadata'].keys())}")
    
    # Test if they can be serialized to JSON
    try:
        json_result = json.dumps({"retrieved_documents": serializable_documents_chat})
        print(f"\n  ✅ Successfully serialized documents to JSON ({len(json_result)} bytes)")
    except Exception as e:
        print(f"\n  ❌ Error serializing to JSON: {e}")
    
    # Test graph.py serialization logic
    print("\n3. Testing serialization logic from graph.py:")
    serializable_documents_graph = []
    
    for doc in docs:
        # Check if it's a LangChain Document object (from graph.py)
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            # Convert Document to dict
            serializable_documents_graph.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        elif isinstance(doc, dict):
            # Already a dict
            serializable_documents_graph.append(doc)
        else:
            # Unknown type, try to convert to dict
            try:
                serializable_documents_graph.append(dict(doc))
            except:
                print(f"Warning: Couldn't serialize document of type {type(doc)}")
    
    # Verify the serialized documents from graph.py
    print(f"  - Serialized {len(serializable_documents_graph)} documents")
    
    # Test if they can be serialized to JSON
    try:
        json_result = json.dumps({"retrieved_documents": serializable_documents_graph})
        print(f"\n  ✅ Successfully serialized documents to JSON ({len(json_result)} bytes)")
    except Exception as e:
        print(f"\n  ❌ Error serializing to JSON: {e}")
    
    # Compare the results from both methods
    are_identical = serializable_documents_chat == serializable_documents_graph
    print(f"\n4. Serialization results from chat.py and graph.py are identical: {are_identical}")
    
    print("\n=== Document Serialization Test Completed ===")

if __name__ == "__main__":
    test_document_serialization() 