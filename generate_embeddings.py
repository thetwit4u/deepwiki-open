import os
import sys

# Add the current directory to the Python path so imports work
sys.path.append('.')

def generate_embeddings(repo_id):
    """Generate embeddings for a repository and store them in ChromaDB."""
    from api.langgraph.graph import run_rag_pipeline
    
    # Get the full path to the repository
    repo_dir = os.path.join(os.path.expanduser('~'), 'Dev', 'ai-projects', 'deepwiki-open', 'wiki-data', 'repos', repo_id)
    
    if not os.path.exists(repo_dir):
        print(f"Repository directory not found: {repo_dir}")
        return False
    
    print(f"Generating embeddings for {repo_id} from {repo_dir}")
    
    # Call run_rag_pipeline with force reindexing
    result = run_rag_pipeline(
        repo_identifier=repo_dir,
        query="What is this repository about?",
        embedding_provider="ollama_nomic",
        generator_provider="gemini",
        skip_indexing=False,  # Force indexing
        debug=True
    )
    
    collection_name = result.get("metadata", {}).get("collection_name", None)
    
    if collection_name:
        print(f"Successfully created collection: {collection_name}")
        print(f"Chunk count: {result.get('metadata', {}).get('chunk_count', 'N/A')}")
        return True
    else:
        print("Failed to create collection")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_embeddings.py <repository_id>")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    success = generate_embeddings(repo_id)
    
    if success:
        print("Embeddings generated successfully.")
    else:
        print("Failed to generate embeddings.")
        sys.exit(1) 