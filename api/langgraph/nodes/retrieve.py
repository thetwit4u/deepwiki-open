from api.langgraph.state import RAGState
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from api.langgraph.chroma_utils import generate_collection_name, get_persistent_dir, get_chroma_client, check_collection_exists
from api.langgraph.embeddings import get_embedding_function
from typing import Dict, Any
import os
import chromadb

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant documents for a query from a vectorstore.
    
    Args:
        state: Dictionary containing the query and vectorstore
    
    Returns:
        Updated state with retrieved documents
    """
    result = state.copy()
    
    # Check if we already have a vectorstore in state
    vectorstore = state.get('vectorstore')
    collection_name = state.get('collection_name')
    query = state.get('query')
    repo_identifier = state.get('repo_identifier')
    embedding_provider = state.get('embedding_provider', 'ollama_nomic')
    top_k = state.get('top_k', 5)
    
    if not query:
        print("No query provided to retrieve_node")
        result["retrieved_documents"] = []
        return result
    
    # Load vectorstore if not provided
    if not vectorstore and repo_identifier:
        print(f"Loading vectorstore for repository: {repo_identifier}")
        
        # Determine if we need a specific collection name
        if not collection_name:
            collection_name = generate_collection_name(repo_identifier)
        
        # Check if collection exists
        persistent_dir = get_persistent_dir()
        client = get_chroma_client(persistent_dir)
        if check_collection_exists(client, collection_name):
            print(f"Collection '{collection_name}' exists, loading...")
            
            # Get embedding function based on provider
            embedding_function = get_embedding_function(
                embedding_provider=embedding_provider
            )
            
            # Create vectorstore with the collection
            vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_function
            )
        else:
            print(f"Collection '{collection_name}' not found!")
            result["retrieved_documents"] = []
            result["error_retrieve"] = f"Collection {collection_name} not found"
            return result
    
    if not vectorstore:
        print("No vectorstore available for retrieval!")
        result["retrieved_documents"] = []
        result["error_retrieve"] = "No vectorstore available"
        return result
    
    # Retrieve relevant documents for query
    try:
        docs = vectorstore.similarity_search(query, k=top_k)
        result["retrieved_documents"] = docs  # Store the documents in the state
        print(f"Retrieved {len(docs)} documents from collection '{collection_name}'")
        print(f"Retrieved a total of {len(docs)} documents")
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        result["retrieved_documents"] = []
        result["error_retrieve"] = str(e)
    
    return result

# Usage Example
if __name__ == "__main__":
    from api.langgraph.nodes.store_vectors import store_vectors_node
    from api.langgraph.nodes.embed_documents import embed_documents_node
    from api.langgraph.nodes.split_text import split_text_node
    from api.langgraph.nodes.load_documents import load_documents_node
    state = RAGState()
    state["repo_identifier"] = "/path/to/repo"
    state["query"] = "What does this repo do?"
    state["vectorstore"] = None
    state["collection_name"] = generate_collection_name(state["repo_identifier"])
    state = retrieve_node(state)
    print(len(state.get("retrieved_documents", []))) 