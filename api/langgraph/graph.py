from langgraph.graph import StateGraph, END, START
from api.langgraph.state import RAGState, ConversationMemory
from api.langgraph.nodes.load_documents import load_documents_node
from api.langgraph.nodes.split_text import split_text_node
from api.langgraph.nodes.embed_documents import embed_documents_node
from api.langgraph.nodes.store_vectors import store_vectors_node
from api.langgraph.nodes.retrieve import retrieve_node
from api.langgraph.nodes.generate import generate_node
from api.langgraph.nodes.memory import memory_node
from api.langgraph.chroma_utils import generate_collection_name, get_persistent_dir, check_collection_exists, get_chroma_client
from langchain_community.vectorstores import Chroma
from api.langgraph.embeddings import get_embedding_function
import time
import os
import traceback
from datetime import datetime
import chromadb
from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any, List
from langchain_chroma import Chroma

# --- Graph Construction ---
def get_rag_graph():
    """Constructs and returns the RAG graph."""
    graph = StateGraph(RAGState)
    graph.add_node("load_documents", load_documents_node)
    graph.add_node("split_text", split_text_node)
    graph.add_node("embed_documents", embed_documents_node)
    graph.add_node("store_vectors", store_vectors_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("memory", memory_node)
    graph.add_edge(START, "load_documents")
    graph.add_edge("load_documents", "split_text")
    graph.add_edge("split_text", "embed_documents")
    graph.add_edge("embed_documents", "store_vectors")
    graph.add_edge("store_vectors", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "memory")
    graph.add_edge("memory", END)
    return graph.compile()

rag_graph = get_rag_graph() # Compile it once

# --- Pipeline Runners ---
def debug_rag_pipeline(state):
    """
    Simplified pipeline for debugging, processes the nodes in sequence.
    This is used for fast iteration and debugging without using LangGraph.
    """
    print("\n==== RETRIEVAL-ONLY PIPELINE START ====")
    print(f"Input state: {state}")
    
    # Cache state for each node
    final_state = state.copy()
    
    # Nodes to run in sequence 
    nodes = ["retrieve", "generate"]
    
    for node_name in nodes:
        try:
            print(f"\n==== Running node: {node_name} ====")
            
            # Call the appropriate node function
            if node_name == "retrieve":
                # Set up node access to use the state
                retrieve_result = retrieve_node(final_state)
                # Update state with node result
                final_state.update(retrieve_result)
                
            elif node_name == "generate":
                # Use result from retrieve as input to generate
                generate_result = generate_node(final_state)
                # Update state with node result
                final_state.update(generate_result)
            
            # Add other nodes as needed
                
        except Exception as e:
            print(f"\n==== UNEXPECTED ERROR in debug_rag_pipeline for node {node_name} ====\nError: {str(e)}")
            import traceback
            traceback.print_exc()
            # Add error to state
            final_state[f"error_{node_name}"] = str(e)
            final_state["answer"] = f"An error occurred in the {node_name} node: {str(e)}"
            break
    
    print("\n==== RETRIEVAL-ONLY PIPELINE COMPLETE ====")
    
    # Convert final_state to return format expected by the API
    # This matches the format from the run_rag_pipeline function
    retrieved_documents = final_state.get("retrieved_documents", [])
    serializable_documents = []
    
    for doc in retrieved_documents:
        # Check if it's a LangChain Document object
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            # Convert Document to dict
            serializable_documents.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        elif isinstance(doc, dict):
            # Already a dict
            serializable_documents.append(doc)
        else:
            # Unknown type, try to convert to dict
            try:
                serializable_documents.append(dict(doc))
            except:
                print(f"Warning: Couldn't serialize document of type {type(doc)}")
    
    # Return the response in the expected format
    return {
        "answer": final_state.get("answer", ""),
        "memory": final_state.get("memory"),
        "retrieved_documents": serializable_documents,
        "metadata": {
            "collection_name": final_state.get("collection_name"),
            "timestamp": datetime.now().isoformat()
        }
    }

def run_retrieval_only_pipeline(state):
    """
    Run only the retrieval and generation parts of the pipeline.
    This is used when we already have an indexed repository and just want to query it.
    
    Args:
        state: Dictionary containing the query state with collection_name
        
    Returns:
        Response with answer and retrieved documents
    """
    print("\n==== RETRIEVAL-ONLY PIPELINE START ====")
    print(f"Input state: {state}")
    
    # Cache state for each node
    final_state = state.copy()
    
    try:
        # Run the retrieve node
        print("\n==== Running node: retrieve ====")
        retrieve_result = retrieve_node(final_state)
        final_state.update(retrieve_result)
        
        # Run the generate node
        print("\n==== Running node: generate ====")
        generate_result = generate_node(final_state)
        final_state.update(generate_result)
        
        print("\n==== RETRIEVAL-ONLY PIPELINE COMPLETE ====")
    except Exception as e:
        print(f"Error in retrieval pipeline: {e}")
        import traceback
        traceback.print_exc()
        final_state["error"] = str(e)
        final_state["answer"] = f"An error occurred: {str(e)}"
    
    # Convert final state to response format
    # This matches the format from run_rag_pipeline
    retrieved_documents = final_state.get("retrieved_documents", [])
    serializable_documents = []
    
    for doc in retrieved_documents:
        # Check if it's a LangChain Document object
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            # Convert Document to dict
            serializable_documents.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        elif isinstance(doc, dict):
            # Already a dict
            serializable_documents.append(doc)
        else:
            # Unknown type, try to convert to dict
            try:
                serializable_documents.append(dict(doc))
            except:
                print(f"Warning: Couldn't serialize document of type {type(doc)}")
    
    # Return the response in the expected format
    return {
        "answer": final_state.get("answer", ""),
        "memory": final_state.get("memory"),
        "retrieved_documents": serializable_documents,
        "metadata": {
            "collection_name": final_state.get("collection_name"),
            "timestamp": datetime.now().isoformat()
        }
    }

def full_rag_pipeline(state):
    """
    Full RAG pipeline: runs all nodes in sequence (indexing + retrieval + generation).
    """
    print("\n==== FULL RAG PIPELINE START ====")
    print(f"Input state: {state}")
    final_state = state.copy()
    nodes = [
        ("load_documents", load_documents_node),
        ("split_text", split_text_node),
        ("embed_documents", embed_documents_node),
        ("store_vectors", store_vectors_node),
        ("retrieve", retrieve_node),
        ("generate", generate_node),
        ("memory", memory_node),
    ]
    for node_name, node_fn in nodes:
        try:
            print(f"\n==== Running node: {node_name} ====")
            result = node_fn(final_state)
            final_state.update(result)
        except Exception as e:
            print(f"\n==== ERROR in full_rag_pipeline for node {node_name} ====\nError: {str(e)}")
            import traceback
            traceback.print_exc()
            final_state[f"error_{node_name}"] = str(e)
            final_state["answer"] = f"An error occurred in the {node_name} node: {str(e)}"
            break
    print("\n==== FULL RAG PIPELINE COMPLETE ====")
    # Convert to response format
    retrieved_documents = final_state.get("retrieved_documents", [])
    serializable_documents = []
    for doc in retrieved_documents:
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            serializable_documents.append({"page_content": doc.page_content, "metadata": doc.metadata})
        elif isinstance(doc, dict):
            serializable_documents.append(doc)
        else:
            try:
                serializable_documents.append(dict(doc))
            except:
                print(f"Warning: Couldn't serialize document of type {type(doc)}")
    return {
        "answer": final_state.get("answer", ""),
        "memory": final_state.get("memory"),
        "retrieved_documents": serializable_documents,
        "metadata": {
            "collection_name": final_state.get("collection_name"),
            "timestamp": datetime.now().isoformat()
        }
    }

def run_rag_pipeline(
    repo_identifier, 
    query,
    generator_provider="gemini",
    embedding_provider="ollama_nomic",
    top_k=10,
    memory=None,
    debug=False,
    skip_indexing=False,
    collection_name=None
):
    """
    Run the RAG pipeline for the given repository identifier and query.
    """
    try:
        vectors_only_mode = False
        if collection_name:
            print(f"Processing query with {collection_name}")
            print(f"Using explicitly provided collection name: {collection_name}")
            client = get_chroma_client(get_persistent_dir())
            if check_collection_exists(client, collection_name):
                print(f"Verified collection '{collection_name}' exists via direct access.")
                vectors_only_mode = True
            else:
                print(f"Warning: Provided collection '{collection_name}' does not exist.")
        if not vectors_only_mode and not skip_indexing:
            if not os.path.isdir(repo_identifier):
                repo_path = os.path.abspath(os.path.join(os.getcwd(), repo_identifier))
                if not os.path.isdir(repo_path):
                    if not collection_name:
                        raise ValueError(f"Local directory does not exist: {repo_identifier}")
                    else:
                        print(f"Repository path not found, but using collection: {collection_name}")
                        vectors_only_mode = True
                else:
                    repo_identifier = repo_path
        state = {
            "query": query,
            "top_k": top_k,
            "use_ollama": generator_provider == "ollama",
            "embedding_provider": embedding_provider,
            "collection_name": collection_name,
            "repo_identifier": repo_identifier
        }
        if memory:
            state["memory"] = memory
        print(f"Initial state: {state}")
        # --- Pipeline selection logic ---
        if not skip_indexing:
            return full_rag_pipeline(state)
        elif debug and vectors_only_mode:
            return run_retrieval_only_pipeline(state)
        elif debug:
            return debug_rag_pipeline(state)
        else:
            return debug_rag_pipeline(state)
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in run_rag_pipeline: {str(e)}\n{error_traceback}")
        return {
            "answer": f"An error occurred: {str(e)}",
            "memory": memory,
            "retrieved_documents": [],
            "metadata": {"error": str(e), "error_traceback": error_traceback, "timestamp": datetime.now().isoformat()},
            "error": True
        }

# Example of how to run
if __name__ == "__main__":
    test_repo = "https://github.com/langchain-ai/langgraph"
    test_query = "What is LangGraph?"
    
    print("--- TESTING WITH OPENAI EMBEDDINGS AND GEMINI GENERATOR ---")
    result_openai_gemini = run_rag_pipeline(
        test_repo, test_query, 
        generator_provider="gemini", 
        embedding_provider="openai", 
        skip_indexing=False, debug=True
    )
    print(f"Answer: {result_openai_gemini.get('answer')[:100]}...")
    print(f"Metadata: {result_openai_gemini.get('metadata')}")

    # Ensure Ollama is running with nomic-embed-text and a generator model (e.g., qwen3:1.7b)
    # ollama pull nomic-embed-text
    # ollama pull qwen3:1.7b 
    print("\n--- TESTING WITH OLLAMA NOMIC EMBEDDINGS AND OLLAMA GENERATOR ---")
    result_ollama_ollama = run_rag_pipeline(
        test_repo, "How are cycles handled in LangGraph?", 
        generator_provider="ollama", 
        embedding_provider="ollama_nomic", 
        skip_indexing=True, # Assuming previous run created the collection with OpenAI embeddings
                            # For a true ollama test, set skip_indexing=False or delete the collection
        debug=True,
        memory=result_openai_gemini.get('memory')
    )
    # Note: If skip_indexing=True and the collection was made with different embeddings,
    # retrieval quality will be poor. For a clean test with Nomic, ensure the collection
    # is built with Nomic embeddings (e.g., set skip_indexing=False on first Nomic run).
    print(f"Answer: {result_ollama_ollama.get('answer')[:100]}...")
    print(f"Metadata: {result_ollama_ollama.get('metadata')}") 