from api.langgraph.state import RAGState
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from api.langgraph.chroma_utils import generate_collection_name, get_persistent_dir
from api.langgraph.embeddings import get_embedding_function

def retrieve_node(state: RAGState) -> RAGState:
    """
    Retrieves relevant documents from ChromaDB based on a query.
    Expects:
    - state['query']: The search query
    - state['vectorstore']: The ChromaDB vectorstore initialized in store_vectors_node
    - state['collection_name']: The collection name for the current repository
    Returns:
    - state['relevant_documents']: List of retrieved documents
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
        class DefaultConfig:
            class Retriever:
                top_k = 20
            retriever = Retriever()
            embedder: MockEmbedderConfig = MockEmbedderConfig(model="text-embedding-3-small", dimensions=256)
            embedder_ollama: MockEmbedderConfig = MockEmbedderConfig(model="nomic-embed-text")
        api_config = DefaultConfig()
    
    query = state.get('query')
    vectorstore = state.get('vectorstore')
    collection_name = state.get('collection_name')
    repositories = state.get('repositories', [])
    repo_identifier = state.get('repo_identifier')
    embedding_provider = state.get('embedding_provider', 'ollama_nomic')
    
    if not query:
        raise ValueError("No 'query' provided in state for retrieval.")
    
    k = state.get('top_k', api_config.retriever.top_k)
    retrieved_docs = []
    
    if vectorstore:
        print(f"Using existing vectorstore for collection '{collection_name}'")
        try:
            retrieved_docs = vectorstore.similarity_search(query, k=k)
            print(f"Retrieved {len(retrieved_docs)} documents from collection '{collection_name}'")
        except Exception as e:
            print(f"Error retrieving from existing vectorstore: {str(e)}")
            vectorstore = None
    
    if not vectorstore and repo_identifier:
        print(f"Loading vectorstore for repository: {repo_identifier}")
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            persistent_dir = get_persistent_dir()
            client = chromadb.PersistentClient(path=persistent_dir)
            
            try:
                import time
                from api.langgraph.chroma_utils import check_collection_exists
                
                exists = check_collection_exists(client, collection_name)
                
                if exists:
                    try:
                        print(f"Collection '{collection_name}' exists, loading...")
                        # Get embedding function based on the provider in state
                        embedding_function = get_embedding_function(embedding_provider, api_config)
                        
                        chroma_collection = client.get_collection(name=collection_name)
                        vectorstore = Chroma(
                            client=client,
                            collection_name=collection_name,
                            embedding_function=embedding_function
                        )
                        state['vectorstore'] = vectorstore
                        state['collection_name'] = collection_name
                        retrieved_docs = vectorstore.similarity_search(query, k=k)
                        print(f"Retrieved {len(retrieved_docs)} documents from collection '{collection_name}'")
                    except Exception as e:
                        print(f"Error accessing collection: {e}")
                        state['relevant_documents'] = []
                        return state
                else:
                    print(f"Collection '{collection_name}' not found, cannot retrieve documents.")
                    state['relevant_documents'] = []
                    return state
            except Exception as e:
                print(f"Error checking collection existence: {e}")
                state['relevant_documents'] = []
                return state
        except Exception as e:
            print(f"Error loading vectorstore for {repo_identifier}: {str(e)}")
            vectorstore = None
            retrieved_docs = []
    
    # Multi-repo search if no specific repo docs found or repositories list provided
    if (not retrieved_docs or not repo_identifier) and repositories:
        print(f"Searching across multiple repositories: {repositories}")
        multi_repo_docs = []
        for repo in repositories:
            try:
                import chromadb
                persistent_dir = get_persistent_dir()
                client = chromadb.PersistentClient(path=persistent_dir)
                repo_collection_name = generate_collection_name(repo)
                
                try:
                    from api.langgraph.chroma_utils import check_collection_exists
                    exists = check_collection_exists(client, repo_collection_name)
                    if not exists:
                        print(f"Collection for {repo} not found, skipping.")
                        continue
                    
                    try:
                        # Get embedding function based on the provider in state
                        embedding_function = get_embedding_function(embedding_provider, api_config)
                        
                        repo_vectorstore = Chroma(
                            client=client,
                            collection_name=repo_collection_name,
                            embedding_function=embedding_function
                        )
                        repo_docs = repo_vectorstore.similarity_search(query, k=k//2)
                        print(f"Retrieved {len(repo_docs)} documents from {repo}")
                        for doc in repo_docs:
                            doc.metadata['repository'] = repo
                        multi_repo_docs.extend(repo_docs)
                    except Exception as e:
                        print(f"Error searching repository {repo}: {e}")
                        continue
                except Exception as e:
                    print(f"Error initializing ChromaDB for {repo}: {e}")
                    continue
            except Exception as e:
                print(f"Error searching repository {repo}: {str(e)}")
        retrieved_docs.extend(multi_repo_docs)
        if len(retrieved_docs) > k:
            retrieved_docs = retrieved_docs[:k]
    
    state['relevant_documents'] = retrieved_docs
    print(f"Retrieved a total of {len(retrieved_docs)} documents")
    return state

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
    print(len(state.get("relevant_documents", []))) 