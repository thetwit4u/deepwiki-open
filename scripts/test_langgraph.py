"""
Test script for the LangGraph RAG pipeline.

Usage:
  python -m api.test_langgraph --repo https://github.com/username/repo
  python -m api.test_langgraph --local /path/to/local/directory
"""

import argparse
import time
import os
from api.langgraph_rag import run_rag_pipeline, ConversationMemory, RAGState, load_documents_node, split_text_node, embed_documents_node, store_vectors_node, retrieve_node, generate_node

def main():
    parser = argparse.ArgumentParser(description="Test LangGraph RAG pipeline")
    
    # Repository specification options
    repo_group = parser.add_mutually_exclusive_group(required=True)
    repo_group.add_argument("--repo", help="GitHub/GitLab/Bitbucket repository URL")
    repo_group.add_argument("--local", help="Local directory path")
    
    # Multi-repository search options
    parser.add_argument("--multi-repo", action="store_true", help="Enable multi-repository search mode")
    parser.add_argument("--add-repo", action="append", help="Additional repositories to include in search (can be used multiple times)")
    
    # Model and retrieval settings
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama models")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    
    # Debug options
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode (use compiled graph)")
    parser.add_argument("--update-collections", action="store_true", help="Update existing collections instead of recreating them")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing even if collection exists")
    
    args = parser.parse_args()
    
    # Set configuration from command line arguments
    if args.update_collections:
        from api.langgraph_config import config
        config.update_existing_collections = True
    
    # Determine repository identifier
    repo_identifier = args.repo if args.repo else args.local
    
    # Convert relative path to absolute if it's a local directory
    if args.local and not args.local.startswith('/'):
        repo_identifier = os.path.abspath(args.local)
    
    # Verify the repository identifier exists
    if args.local and not os.path.exists(repo_identifier):
        print(f"❌ Local directory not found: {repo_identifier}")
        return
    
    # Process additional repositories for multi-repo search
    additional_repos = []
    if args.add_repo:
        for repo in args.add_repo:
            # Convert local paths to absolute
            if not (repo.startswith('http://') or repo.startswith('https://')):
                repo = os.path.abspath(repo)
                if not os.path.exists(repo):
                    print(f"⚠️ Warning: Additional repository not found: {repo}")
                    continue
            additional_repos.append(repo)
            print(f"Added repository for multi-search: {repo}")
    
    # Simple command-line chat interface
    memory = ConversationMemory()
    
    print(f"\n🔍 DeepWiki LangGraph RAG Demo")
    print(f"Primary Repository: {repo_identifier}")
    print(f"Multi-Repository Mode: {'Enabled' if args.multi_repo or additional_repos else 'Disabled'}")
    if additional_repos:
        print(f"Additional Repositories: {len(additional_repos)}")
    print(f"Using {'Ollama' if args.ollama else 'OpenAI/Gemini'} models")
    print(f"Debug Mode: {'Disabled' if args.no_debug else 'Enabled'}")
    print(f"Skip Indexing: {'Disabled' if args.force_reindex else 'Enabled'}")
    print("\nInitializing repository... This may take a moment.\n")
    
    # Pre-initialize the repo (warm-up)
    try:
        print("🔍 Initializing repository...")
        # Perform a simple query to initialize the repository
        result = run_rag_pipeline(
            repo_identifier=repo_identifier,
            query="What is this repository about?",
            use_ollama=args.ollama,
            top_k=1,  # Minimal retrieval for initialization
            memory=memory,
            debug=not args.no_debug,
            skip_indexing=False  # Always index on first run
        )
        print("✅ Repository initialized successfully!\n")
    except Exception as e:
        print(f"⚠️ Initialization warning: {str(e)}")
        print("Continuing anyway...\n")
    
    try:
        while True:
            # Get user input
            query = input("\n💬 User: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            # Special command to toggle multi-repo mode
            if query.lower() == "!multi":
                args.multi_repo = not args.multi_repo
                print(f"Multi-repository mode {'enabled' if args.multi_repo else 'disabled'}")
                continue
            
            # Special command to toggle indexing mode
            if query.lower() == "!reindex":
                args.force_reindex = not args.force_reindex
                print(f"Force reindexing {'enabled' if args.force_reindex else 'disabled'}")
                continue

            # Special command to show a help menu
            if query.lower() in ["!help", "!commands"]:
                print("\nAvailable commands:")
                print("  !help, !commands - Show this help menu")
                print("  !multi - Toggle multi-repository mode")
                print("  !reindex - Toggle force reindexing mode")
                print("  !repos - Show all repositories being searched")
                print("  !settings - Show current settings")
                print("  exit, quit, q - Exit the program")
                continue
                
            # Show repositories
            if query.lower() == "!repos":
                print(f"\nPrimary repository: {repo_identifier}")
                if additional_repos:
                    print("Additional repositories:")
                    for i, repo in enumerate(additional_repos, 1):
                        print(f"  {i}. {repo}")
                continue
                
            # Show settings
            if query.lower() == "!settings":
                print("\nCurrent settings:")
                print(f"  Multi-repository mode: {'Enabled' if args.multi_repo else 'Disabled'}")
                print(f"  Model: {'Ollama' if args.ollama else 'OpenAI/Gemini'}")
                print(f"  Top-k documents: {args.top_k}")
                print(f"  Debug mode: {'Disabled' if args.no_debug else 'Enabled'}")
                print(f"  Update collections: {args.update_collections}")
                print(f"  Force reindexing: {args.force_reindex}")
                continue
            
            # Time the response
            start_time = time.time()
            
            # Run the RAG pipeline
            print("\n⏳ Processing...")
            try:
                # Set up repositories for multi-repo search if enabled
                repos_to_search = None
                if args.multi_repo and additional_repos:
                    repos_to_search = [repo_identifier] + additional_repos
                
                # Run the pipeline
                result = run_rag_pipeline(
                    repo_identifier=repo_identifier,
                    query=query,
                    use_ollama=args.ollama,
                    top_k=args.top_k,
                    memory=memory,
                    debug=not args.no_debug,
                    repositories=repos_to_search if args.multi_repo else None,
                    skip_indexing=not args.force_reindex  # Skip indexing if not forcing reindex
                )
                
                # Update memory for next iteration
                memory = result["memory"]
                
                # Print the result
                elapsed_time = time.time() - start_time
                print(f"\n🤖 Assistant ({elapsed_time:.2f}s):")
                print(result["answer"])
                
                # Optionally show retrieved documents
                if input("\nShow retrieved documents? [y/N]: ").lower() == "y":
                    print("\n📄 Retrieved Documents:")
                    for i, doc in enumerate(result.get("retrieved_documents", [])):
                        file_path = doc.metadata.get('file_path', 'Unknown file')
                        repo_name = doc.metadata.get('repository', 'Current repo')
                        print(f"\n--- Document {i+1} from {file_path} ({repo_name}) ---")
                        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    
                # Optionally show metadata
                if "metadata" in result and input("\nShow metadata? [y/N]: ").lower() == "y":
                    print("\n📊 Metadata:")
                    metadata = result["metadata"]
                    for key, value in metadata.items():
                        if key != "error_traceback":  # Skip lengthy tracebacks
                            print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    print("\nThank you for using DeepWiki LangGraph RAG!")

def test_load_documents_only(repo_identifier):
    state = RAGState.from_dict({"repo_identifier": repo_identifier})
    result = load_documents_node(state)
    docs = result.get("documents", [])
    print(f"\nLoaded {len(docs)} documents.")
    if docs:
        print("Sample document metadata:")
        for doc in docs[:3]:
            print(doc.metadata)
    else:
        print("No documents loaded.")

def test_split_text_only(repo_identifier):
    state = RAGState.from_dict({"repo_identifier": repo_identifier})
    state = load_documents_node(state)
    state = split_text_node(state)
    chunks = state.get("chunks", [])
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print("Sample chunk metadata and preview:")
        for chunk in chunks[:3]:
            print(chunk.metadata)
            print("Chunk preview:", chunk.page_content[:100])
    else:
        print("No chunks generated.")

def test_embed_documents_only(repo_identifier):
    state = RAGState.from_dict({"repo_identifier": repo_identifier})
    state = load_documents_node(state)
    state = split_text_node(state)
    state = embed_documents_node(state)
    chunks = state.get("chunks", [])
    embeddings = state.get("embeddings", [])
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Total embeddings: {len(embeddings)}")
    if chunks and embeddings:
        print("Sample chunk metadata and embedding length:")
        for chunk in chunks[:3]:
            emb = chunk.metadata.get('embedding')
            print(chunk.metadata)
            print("Embedding length:", len(emb) if emb is not None else 'None')
            print("Chunk preview:", chunk.page_content[:100])
    else:
        print("No embeddings generated.")

def test_store_vectors_only(repo_identifier):
    state = RAGState.from_dict({"repo_identifier": repo_identifier})
    state = load_documents_node(state)
    state = split_text_node(state)
    state = embed_documents_node(state)
    state = store_vectors_node(state)
    collection_name = state.get("collection_name")
    chunks = state.get("chunks", [])
    print(f"\nCollection name: {collection_name}")
    print(f"Total vectors stored: {len(chunks)}")
    if chunks:
        print("Sample stored chunk metadata:")
        for chunk in chunks[:3]:
            print(chunk.metadata)
    else:
        print("No vectors stored.")

def test_retrieve_only(repo_identifier, query="What is this repository about?"):
    state = RAGState.from_dict({"repo_identifier": repo_identifier, "query": query})
    state = load_documents_node(state)
    state = split_text_node(state)
    state = embed_documents_node(state)
    state = store_vectors_node(state)
    state = retrieve_node(state)
    retrieved = state.get("relevant_documents", [])
    print(f"\nRetrieved {len(retrieved)} documents for query: '{query}'")
    if retrieved:
        print("Sample retrieved document metadata and preview:")
        for doc in retrieved[:3]:
            print(doc.metadata)
            print("Content preview:", doc.page_content[:100])
    else:
        print("No documents retrieved.")

def test_generate_only(repo_identifier, query="What is this repository about?"):
    state = RAGState.from_dict({"repo_identifier": repo_identifier, "query": query})
    state = load_documents_node(state)
    state = split_text_node(state)
    state = embed_documents_node(state)
    state = store_vectors_node(state)
    state = retrieve_node(state)
    state = generate_node(state)
    answer = state.get("answer", "<No answer generated>")
    print(f"\nGenerated answer for query: '{query}'\n{'='*40}\n{answer}\n{'='*40}")
    # Optionally print a summary of retrieved docs
    retrieved = state.get("relevant_documents", [])
    print(f"Retrieved {len(retrieved)} documents.")
    if retrieved:
        print("Sample retrieved document metadata:")
        for doc in retrieved[:2]:
            print(doc.metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LangGraph full pipeline: document loading, splitting, embedding, vector storage, retrieval, and answer generation")
    parser.add_argument("--repo", required=True, help="Repository URL or local path")
    parser.add_argument("--test-split", action="store_true", help="Test document loading and text splitting only")
    parser.add_argument("--test-embed", action="store_true", help="Test document loading, splitting, and embedding")
    parser.add_argument("--test-store", action="store_true", help="Test document loading, splitting, embedding, and vector storage")
    parser.add_argument("--test-retrieve", action="store_true", help="Test document loading, splitting, embedding, vector storage, and retrieval")
    parser.add_argument("--test-generate", action="store_true", help="Test full pipeline including answer generation")
    parser.add_argument("--query", type=str, default="What is this repository about?", help="Query for retrieval/generation test")
    args = parser.parse_args()
    if args.test_generate:
        test_generate_only(args.repo, args.query)
    elif args.test_retrieve:
        test_retrieve_only(args.repo, args.query)
    elif args.test_store:
        test_store_vectors_only(args.repo)
    elif args.test_embed:
        test_embed_documents_only(args.repo)
    elif args.test_split:
        test_split_text_only(args.repo)
    else:
        main() 