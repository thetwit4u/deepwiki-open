from api.langgraph.state import RAGState
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from api.langgraph.chroma_utils import generate_collection_name, get_persistent_dir, get_chroma_client
from api.langgraph.embeddings import get_embedding_function
from langchain_community.embeddings import OllamaEmbeddings

def store_vectors_node(state: RAGState) -> RAGState:
    """
    Stores document chunks and their embeddings in a persistent ChromaDB collection.
    Each repository gets its own collection.
    Uses the embedding function specified by 'embedding_provider' in the RAGState.
    """
    try:
        from api.langgraph_config import config as api_config
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class MockEmbedderConfig:
            model: str
            dimensions: int = None
        @dataclass
        class MockApiConfig:
            embedder: MockEmbedderConfig = MockEmbedderConfig(model="text-embedding-3-small", dimensions=256)
            embedder_ollama: MockEmbedderConfig = MockEmbedderConfig(model="nomic-embed-text")
            update_existing_collections: bool = False
            class Retriever:
                top_k = 10 
            retriever = Retriever()
        api_config = MockApiConfig()

    chunks = state.get('chunks', [])
    embeddings = state.get('embeddings', [])
    repo_identifier = state.get('repo_identifier')
    embedding_provider = state.get('embedding_provider', 'ollama_nomic')

    if not chunks or not repo_identifier:
        raise ValueError("Missing required data (chunks or repo_identifier) for vector storage.")

    # Use collection_name from state if provided, otherwise generate it
    collection_name = state.get('collection_name')
    if not collection_name:
        collection_name = generate_collection_name(repo_identifier)
    
    persistent_dir = get_persistent_dir()
    print(f"Storing vectors in ChromaDB collection '{collection_name}' at {persistent_dir} using {embedding_provider}")

    embedding_function = get_embedding_function(embedding_provider, api_config)
    
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            import chromadb
            import time
            import shutil
            import os
            import random
            client = get_chroma_client(persistent_dir)
            force_recreate = False
            try:
                collections = client.list_collections()
                exists = any(c.name == collection_name for c in collections)
            except KeyError as e:
                if "'_type'" in str(e):
                    print(f"ChromaDB format error detected: {e}")
                    print("Will attempt to recreate collection...")
                    force_recreate = True
                    exists = False
                else:
                    raise e
            except Exception as e:
                print(f"Error listing collections: {e}")
                print("Will attempt to recreate collection...")
                force_recreate = True
                exists = False
            
            recreate = force_recreate or not api_config.update_existing_collections
            
            if recreate and exists:
                print(f"Recreating collection '{collection_name}' with {embedding_provider} embeddings")
                try:
                    client.delete_collection(name=collection_name)
                    print(f"Deleted existing collection '{collection_name}'")
                    time.sleep(0.5)  # Small delay to ensure deletion completes
                    exists = False
                except Exception as del_error:
                    print(f"Error deleting collection: {del_error}")
                    # Continue even if deletion fails; we'll try to create it anyway
            
            try:
                if exists and not force_recreate:
                    print(f"Using existing collection '{collection_name}'")
                    try:
                        chroma_collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
                    except Exception as e:
                        print(f"Error accessing existing collection: {e}")
                        exists = False
                
                if not exists or force_recreate:
                    print(f"Creating new collection '{collection_name}'")
                    try:
                        chroma_collection = client.create_collection(name=collection_name, embedding_function=embedding_function)
                    except chromadb.errors.UniqueConstraintError:
                        print(f"Collection '{collection_name}' already exists during creation")
                        time.sleep(0.5)
                        try:
                            chroma_collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
                        except Exception as e:
                            raise ValueError(f"Failed to access collection after creation attempt: {e}")
                    except Exception as create_error:
                        raise ValueError(f"Failed to create collection: {create_error}")
                
                vectorstore = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embedding_function
                )
                doc_embeddings_list = state.get('embeddings', None)

                if doc_embeddings_list is None or len(doc_embeddings_list) != len(chunks):
                    print("Warning: state['embeddings'] not found or mismatched. Attempting to extract from chunk metadata.")
                    doc_embeddings_list = []
                    temp_chunks_for_metadata = []
                    for chunk in chunks:
                        meta_copy = dict(chunk.metadata)
                        embedding_vector = meta_copy.pop('embedding', None)
                        if embedding_vector is None:
                            raise ValueError(f"Embedding not found in chunk metadata for chunk: {chunk.page_content[:50]}...")
                        doc_embeddings_list.append(embedding_vector)
                        temp_chunks_for_metadata.append(type('Chunk', (), {'metadata': meta_copy, 'page_content': chunk.page_content})())
                    
                    metadatas = [c.metadata for c in temp_chunks_for_metadata]
                    texts = [c.page_content for c in temp_chunks_for_metadata]
                else:
                    metadatas = []
                    for chunk in chunks:
                        meta_copy = dict(chunk.metadata)
                        meta_copy.pop('embedding', None)
                        metadatas.append(meta_copy)
                    texts = [chunk.page_content for chunk in chunks]

                ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
                try:
                    # Ensure embeddings are in the format ChromaDB expects (list of lists of floats)
                    # Sometimes embeddings can be numpy arrays or other formats
                    processed_embeddings = []
                    for emb in doc_embeddings_list:
                        # Convert numpy arrays to lists if needed
                        if hasattr(emb, 'tolist'):
                            processed_embeddings.append(emb.tolist())
                        else:
                            # Ensure it's a list of floats
                            processed_embeddings.append([float(e) for e in emb])
                    
                    print(f"Adding {len(chunks)} documents to ChromaDB collection '{collection_name}'")
                    print(f"First embedding vector sample: {processed_embeddings[0][:5]}... (length: {len(processed_embeddings[0])})")
                    
                    chroma_collection.add(
                        embeddings=processed_embeddings,
                        metadatas=metadatas,
                        documents=texts,
                        ids=ids
                    )
                    print(f"Successfully stored {len(chunks)} documents in ChromaDB collection '{collection_name}'")
                    state['vectorstore'] = vectorstore
                    state['collection_name'] = collection_name
                    break
                except Exception as e:
                    print(f"Error adding vectors to collection: {e}")
                    if retry_count == max_retries - 1:
                        raise ValueError(f"Failed to add vectors to collection after {max_retries} attempts: {e}")
                    retry_count += 1
                    print(f"Retrying vector storage (attempt {retry_count}/{max_retries})...")
                    time.sleep(1)
                    continue
            except Exception as e:
                print(f"Error working with collection: {e}")
                if retry_count == max_retries - 1:
                    raise ValueError(f"Failed to work with collection after {max_retries} attempts: {e}")
                retry_count += 1
                print(f"Retrying collection operations (attempt {retry_count}/{max_retries})...")
                time.sleep(1)
                continue
        except Exception as e:
            print(f"Error in store_vectors_node: {e}")
            if retry_count == max_retries - 1:
                raise ValueError(f"Failed to initialize ChromaDB after multiple attempts: {e}")
            retry_count += 1
            print(f"Retrying entire process (attempt {retry_count}/{max_retries})...")
            time.sleep(1)
    return state

# Usage Example
if __name__ == "__main__":
    from api.langgraph.nodes.embed_documents import embed_documents_node
    from api.langgraph.nodes.split_text import split_text_node
    from api.langgraph.nodes.load_documents import load_documents_node
    from api.langgraph_config import config as api_config_main
    
    # Mock RAGState for the example
    state = RAGState()
    state["repo_identifier"] = "./test_repo_store"
    state["query"] = "test query"
    state["embedding_provider"] = "ollama_nomic"

    # Create a dummy test_repo_store directory with a file
    import os
    if not os.path.exists("./test_repo_store"):
        os.makedirs("./test_repo_store")
    with open("./test_repo_store/sample.txt", "w") as f:
        f.write("This is a sample document for testing vector storage.")

    print("--- Running Test: Load Documents ---")
    state = load_documents_node(state)
    print(f"Documents loaded: {len(state.get('documents', []))}")
    
    print("--- Running Test: Split Text ---")
    state = split_text_node(state)
    print(f"Chunks created: {len(state.get('chunks', []))}")
    
    print(f"--- Running Test: Embed Documents (Provider: {state['embedding_provider']}) ---")
    state = embed_documents_node(state)
    print(f"Embeddings generated: {len(state.get('embeddings', []))}")
    if state.get('embeddings'):
        print(f"First embedding vector dimension: {len(state.get('embeddings')[0])}")

    print("--- Running Test: Store Vectors ---")
    state = store_vectors_node(state)
    
    print(f"\n--- Test Complete ---")
    print(f"Collection name: {state.get('collection_name')}")
    print(f"Vector store object: {state.get('vectorstore')}")

    # Clean up dummy repo and chromadb data
    import shutil
    if os.path.exists("./test_repo_store"):
        shutil.rmtree("./test_repo_store")
    print("Test finished. Check logs for details.") 