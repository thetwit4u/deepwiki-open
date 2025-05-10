from api.langgraph.state import RAGState
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from api.langgraph.chroma_utils import generate_collection_name, get_persistent_dir

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
        from api.langgraph_config import config
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class DefaultConfig:
            class Retriever:
                top_k = 20
            retriever = Retriever()
        config = DefaultConfig()
    query = state.get('query')
    vectorstore = state.get('vectorstore')
    collection_name = state.get('collection_name')
    repositories = state.get('repositories', [])
    repo_identifier = state.get('repo_identifier')
    if not query:
        raise ValueError("No 'query' provided in state for retrieval.")
    k = state.get('top_k', config.retriever.top_k)
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
        try:
            print(f"Loading vectorstore for repository: {repo_identifier}")
            collection_name = generate_collection_name(repo_identifier)
            persistent_dir = get_persistent_dir()
            import chromadb
            client = chromadb.PersistentClient(path=persistent_dir)
            try:
                collections = []
                try:
                    collections = client.list_collections()
                except KeyError as e:
                    if "'_type'" in str(e):
                        print(f"ChromaDB format error detected: {e}")
                        print("Cannot retrieve from corrupted database. Please reindex the repository.")
                        state['relevant_documents'] = []
                        return state
                    else:
                        raise e
                except Exception as e:
                    print(f"Error listing collections: {e}")
                    state['relevant_documents'] = []
                    return state
                exists = any(c.name == collection_name for c in collections)
                if exists:
                    try:
                        print(f"Collection '{collection_name}' exists, loading...")
                        chroma_collection = client.get_collection(name=collection_name)
                        vectorstore = Chroma(
                            client=client,
                            collection_name=collection_name,
                            embedding_function=OpenAIEmbeddings()
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
    if repositories and len(repositories) > 0:
        print(f"Performing multi-repository search across {len(repositories)} repositories")
        multi_repo_docs = []
        import chromadb
        for repo in repositories:
            try:
                if repo == repo_identifier:
                    continue
                repo_collection_name = generate_collection_name(repo)
                print(f"Searching repository: {repo} (collection: {repo_collection_name})")
                try:
                    client = chromadb.PersistentClient(path=get_persistent_dir())
                    try:
                        collections = client.list_collections()
                        exists = any(c.name == repo_collection_name for c in collections)
                    except KeyError as e:
                        if "'_type'" in str(e):
                            print(f"ChromaDB format error detected for {repo}: {e}")
                            print(f"Skipping repository {repo}")
                            continue
                        else:
                            raise e
                    except Exception as e:
                        print(f"Error listing collections for {repo}: {e}")
                        continue
                    if not exists:
                        print(f"Collection '{repo_collection_name}' not found, skipping...")
                        continue
                    try:
                        repo_vectorstore = Chroma(
                            client=client,
                            collection_name=repo_collection_name,
                            embedding_function=OpenAIEmbeddings()
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