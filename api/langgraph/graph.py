from langgraph.graph import StateGraph, END, START
from api.langgraph.state import RAGState, ConversationMemory
from api.langgraph.nodes.load_documents import load_documents_node
from api.langgraph.nodes.split_text import split_text_node
from api.langgraph.nodes.embed_documents import embed_documents_node
from api.langgraph.nodes.store_vectors import store_vectors_node
from api.langgraph.nodes.retrieve import retrieve_node
from api.langgraph.nodes.generate import generate_node
from api.langgraph.nodes.memory import memory_node
from api.langgraph.chroma_utils import generate_collection_name, get_persistent_dir, check_collection_exists
from langchain_community.vectorstores import Chroma
from api.langgraph.embeddings import get_embedding_function
import time
import os
import traceback
from datetime import datetime
import chromadb
from langchain_openai import OpenAIEmbeddings

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
def debug_rag_pipeline(state: RAGState) -> dict:
    """Simplified pipeline for debugging, runs nodes sequentially."""
    result_state = RAGState.from_dict(state.to_dict())
    print(f"\n==== DEBUG PIPELINE START ====\nInput state: {result_state}")
    node_sequence = [
        ("load_documents", load_documents_node),
        ("split_text", split_text_node),
        ("embed_documents", embed_documents_node),
        ("store_vectors", store_vectors_node),
        ("retrieve", retrieve_node),
        ("generate", generate_node),
        ("memory", memory_node)
    ]
    for node_name, node_func in node_sequence:
        try:
            print(f"\n==== Running node: {node_name} ====")
            start_time = time.time()
            try:
                result_state = node_func(result_state)
                elapsed_time = time.time() - start_time
                print(f"Node {node_name} completed in {elapsed_time:.2f} seconds")
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"\n==== ERROR in node {node_name} after {elapsed_time:.2f} seconds ====\nError: {str(e)}")
                traceback.print_exc()
                result_state[f"error_{node_name}"] = str(e)
                result_state[f"error_traceback_{node_name}"] = traceback.format_exc()
                if node_name in ["load_documents", "split_text", "embed_documents", "store_vectors"]:
                    print(f"Critical node {node_name} failed. Pipeline cannot continue.")
                    break
                else:
                    print(f"Non-critical node {node_name} failed. Attempting to continue pipeline.")
                continue
            if node_name == "load_documents" and "documents" in result_state:
                print(f"Loaded {len(result_state['documents'])} documents")
            elif node_name == "split_text" and "chunks" in result_state:
                print(f"Created {len(result_state['chunks'])} chunks")
        except Exception as e:
            print(f"\n==== UNEXPECTED ERROR in debug_rag_pipeline for node {node_name} ====\nError: {str(e)}")
            traceback.print_exc()
    print(f"\n==== DEBUG PIPELINE COMPLETE ====")
    return result_state

def run_retrieval_only_pipeline(state: RAGState, collection_name: str, api_config) -> RAGState:
    """Simplified pipeline that skips indexing, for existing collections."""
    result_state = RAGState.from_dict(state.to_dict())
    print(f"\n==== RETRIEVAL-ONLY PIPELINE START ====\nInput state: {result_state}")
    try:
        persistent_dir = get_persistent_dir()
        client = chromadb.PersistentClient(path=persistent_dir)
        
        # Check if the collection exists using the check_collection_exists utility
        if not check_collection_exists(client, collection_name):
            print(f"Collection '{collection_name}' not found. Cannot proceed with retrieval-only mode.")
            raise ValueError(f"Collection '{collection_name}' not found. Please generate the wiki first.")
        
        embedding_provider = state.get('embedding_provider', 'ollama_nomic')
        print(f"Using {embedding_provider} embeddings for retrieval")
        
        # Get the embedding function based on the provider
        embedding_function = get_embedding_function(embedding_provider, api_config)
        
        # Setup the vectorstore with the specified collection and embedding function
        print(f"Loading ChromaDB collection '{collection_name}' for retrieval")
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        
        result_state["vectorstore"] = vectorstore
        result_state["collection_name"] = collection_name
        
        # Proceed with the retrieval and generation nodes
        print("\n==== Running node: retrieve ====")
        result_state = retrieve_node(result_state)
        
        print("\n==== Running node: generate ====")
        result_state = generate_node(result_state)
        
        if "memory" in result_state:
            print("\n==== Running node: memory ====")
            result_state = memory_node(result_state)
        
        print("\n==== RETRIEVAL-ONLY PIPELINE COMPLETE ====")
        return result_state
    except Exception as e:
        print(f"Error in retrieval-only pipeline: {e}")
        traceback.print_exc()
        result_state["error_retrieval"] = str(e)
        return result_state

def run_rag_pipeline(
    repo_identifier: str,
    query: str,
    generator_provider: str = "gemini", # gemini, openai, ollama
    embedding_provider: str = "ollama_nomic", # openai, ollama_nomic
    top_k: int = None,
    memory = None,
    debug: bool = True,
    repositories: list = None,
    skip_indexing: bool = False,
    collection_name: str = None
) -> dict:
    """Main entry point to run the RAG pipeline."""
    try:
        from api.langgraph_config import config as api_config # Renamed for clarity
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class MockEmbedderConfig:
            model: str
            dimensions: int = None
        @dataclass
        class MockGeneratorConfig:
            model: str
            temperature: float = 0.7
            top_p: float = 0.8
        @dataclass
        class MockApiConfig:
            embedder: MockEmbedderConfig = MockEmbedderConfig(model="text-embedding-3-small", dimensions=256)
            embedder_ollama: MockEmbedderConfig = MockEmbedderConfig(model="nomic-embed-text", dimensions=768)
            generator: MockGeneratorConfig = MockGeneratorConfig(model="gemini-2.5-flash-preview-04-17")
            generator_ollama: MockGeneratorConfig = MockGeneratorConfig(model="qwen3:1.7b")
            class Retriever:
                top_k = 20
            retriever = Retriever()
        api_config = MockApiConfig()

    # Check if we have a valid repo_identifier or a valid collection_name
    if not repo_identifier and not collection_name:
        raise ValueError("Either repository_identifier or collection_name must be provided")
    
    # Only validate repo_identifier path if it's provided and we're not using a predefined collection
    if repo_identifier and not (collection_name and skip_indexing):
        if not (repo_identifier.startswith("http://") or repo_identifier.startswith("https://")):
            repo_identifier = os.path.abspath(os.path.expanduser(repo_identifier))
            if not os.path.exists(repo_identifier):
                raise ValueError(f"Local directory does not exist: {repo_identifier}")
                
    if repositories:
        repositories = [os.path.abspath(os.path.expanduser(r)) if not (r.startswith("http://") or r.startswith("https://")) else r for r in repositories]
    
    print(f"Processing query with {collection_name if collection_name else repo_identifier}")
    start_time = time.time()
    default_top_k = api_config.retriever.top_k
    collection_exists = False
    
    # Use provided collection_name if available, otherwise generate it
    if collection_name is None and repo_identifier:
        collection_name = generate_collection_name(repo_identifier)
    elif collection_name:
        print(f"Using explicitly provided collection name: {collection_name}")

    if skip_indexing:
        persistent_dir = get_persistent_dir()
        try:
            client = chromadb.PersistentClient(path=persistent_dir)
            
            # Try to directly access the collection first (most reliable)
            try:
                client.get_collection(collection_name)
                collection_exists = True
                print(f"Verified collection '{collection_name}' exists via direct access.")
            except Exception as direct_e:
                if "does not exist" in str(direct_e):
                    collection_exists = False
                else:
                    # Fall back to listing collections
                    try:
                        collections = client.list_collections()
                        # Handle different versions of ChromaDB
                        try:
                            # For ChromaDB <0.6.0
                            collection_exists = any(c.name == collection_name for c in collections)
                        except AttributeError:
                            # For ChromaDB >=0.6.0
                            collection_exists = collection_name in collections
                    except Exception as list_e:
                        print(f"Error listing collections: {list_e}")
                        collection_exists = False
            
            if collection_exists:
                print(f"Collection '{collection_name}' exists. Skipping indexing.")
            else:
                print(f"Collection '{collection_name}' not found. Performing full indexing.")
                skip_indexing = False
        except Exception as e:
            print(f"Error checking collection: {e}. Performing full indexing.")
            skip_indexing = False

    initial_state_dict = {
        "query": query,
        "top_k": top_k or default_top_k,
        "use_ollama": generator_provider == "ollama", # For the generator node
        "embedding_provider": embedding_provider,
        "collection_name": collection_name  # Always include collection_name in state
    }
    
    # Only add repo_identifier to state if it was provided
    if repo_identifier:
        initial_state_dict["repo_identifier"] = repo_identifier
        
    if memory: initial_state_dict["memory"] = memory
    if repositories: initial_state_dict["repositories"] = repositories
    
    state = RAGState.from_dict(initial_state_dict)
    print(f"Initial state: {state}")
    final_state = {}
    try:
        if skip_indexing and collection_exists:
             final_state = run_retrieval_only_pipeline(state, collection_name, api_config)
        elif debug:
            final_state = debug_rag_pipeline(state)
        else:
            final_state = rag_graph.invoke(state)
        
        elapsed_time = time.time() - start_time
        
        # Determine generator model used based on provider
        gen_model_name = "unknown_generator"
        if generator_provider == "gemini":
            gen_model_name = api_config.generator.model
        elif generator_provider == "ollama":
            gen_model_name = api_config.generator_ollama.model
        elif generator_provider == "openai":
            gen_model_name = "gpt- E.g. gpt-4" # Needs specific openai generator model from config
            # Assuming you might add api_config.generator_openai.model later

        emb_model_name = "unknown_embedding"
        if embedding_provider == "openai":
            emb_model_name = api_config.embedder.model
        elif embedding_provider == "ollama_nomic":
            emb_model_name = api_config.embedder_ollama.model

        response = {
            "answer": final_state.get("answer", ""),
            "memory": final_state.get("memory"),
            "retrieved_documents": final_state.get("relevant_documents", []),
            "metadata": {
                "elapsed_time": elapsed_time,
                "repo_identifier": repo_identifier,
                "collection_name": final_state.get("collection_name"),
                "generator_provider": generator_provider,
                "generator_model": gen_model_name,
                "embedding_provider": embedding_provider,
                "embedding_model": emb_model_name,
                "top_k": top_k or default_top_k,
                "timestamp": datetime.now().isoformat(),
                "document_count": len(final_state.get("documents", [])) if "documents" in final_state else None,
                "chunk_count": len(final_state.get("chunks", [])) if "chunks" in final_state else None,
                "multi_repo_mode": bool(repositories),
                "skipped_indexing": skip_indexing and collection_exists,
                "vectors_only": repo_identifier is None and collection_name is not None
            }
        }
        errors = {k: v for k, v in final_state.items() if k.startswith("error_")}
        if errors: response["metadata"]["errors"] = errors
        return response
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