import os
import sys
sys.path.append('.')

from api.langgraph.graph import run_rag_pipeline
from api.langgraph.chroma_utils import generate_collection_name, get_chroma_client, get_persistent_dir
from api.langgraph.wiki_structure import normalize_repo_id

def test_wiki_embeddings(repo_id: str):
    """Test wiki embedding generation with the correct embedding provider.
    
    Args:
        repo_id: Repository ID to test
    """
    # Normalize the repository ID to ensure consistent naming
    normalized_repo_id = normalize_repo_id(repo_id)
    print(f"Repository ID: {repo_id}")
    print(f"Normalized repository ID: {normalized_repo_id}")
    
    # Get the repository path
    repo_dir = os.path.join(os.path.expanduser('~'), 'Dev', 'ai-projects', 'deepwiki-open', 'wiki-data', 'repos', normalized_repo_id)
    
    if not os.path.exists(repo_dir):
        print(f"Repository directory not found: {repo_dir}")
        print("Please make sure the repository exists in the wiki-data/repos directory.")
        return False
    
    print(f"Repository directory: {repo_dir}")
    
    # Generate collection name
    collection_name = generate_collection_name(normalized_repo_id)
    print(f"Collection name: {collection_name}")
    
    # Check if collection already exists
    persistent_dir = get_persistent_dir()
    client = get_chroma_client(persistent_dir)
    
    try:
        collections = client.list_collections()
        if collection_name in collections:
            print(f"Collection {collection_name} already exists. Deleting it to test fresh generation.")
            client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error checking collection: {e}")
    
    # Generate embeddings with ollama_nomic
    print(f"\n===== Generating embeddings for {normalized_repo_id} =====")
    print(f"Using embedding provider: ollama_nomic")
    print(f"Collection name: {collection_name}")
    
    try:
        result = run_rag_pipeline(
            repo_identifier=repo_dir,
            query="What is this repository about?",
            embedding_provider="ollama_nomic",
            generator_provider="gemini",
            skip_indexing=False,  # Force indexing
            debug=True
        )
        
        # Check the result
        if "metadata" in result and "collection_name" in result["metadata"]:
            created_collection = result["metadata"]["collection_name"]
            embedding_provider = result["metadata"]["embedding_provider"]
            print(f"\n===== Results =====")
            print(f"Successfully created collection: {created_collection}")
            print(f"Using embedding provider: {embedding_provider}")
            
            # Verify the embedding provider
            if embedding_provider == "ollama_nomic":
                print("\n✅ SUCCESS: Wiki generation is using ollama_nomic embeddings as expected!")
            else:
                print(f"\n❌ ERROR: Wiki generation is using {embedding_provider} embeddings instead of ollama_nomic!")
                return False
            
            # Print embedding model information
            embedding_model = result["metadata"].get("embedding_model", "unknown")
            print(f"Embedding model: {embedding_model}")
            
            # Check chunk count
            chunk_count = result["metadata"].get("chunk_count", "N/A")
            print(f"Chunk count: {chunk_count}")
            
            # Print answer from query for verification
            if "answer" in result:
                print(f"\nSample query answer:")
                print(f"{result['answer']}")
            
            return True
        else:
            print("Failed to create collection")
            print(f"Result metadata: {result.get('metadata', {})}")
            return False
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: Exception during embedding generation: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Test with the provided repository or default to customs exchange rate
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    else:
        repo_id = "customs.exchange-rate-main"
    
    print(f"\n===== Testing Wiki Embeddings =====")
    print(f"This test verifies that wiki generation consistently uses ollama_nomic embeddings.")
    print(f"Testing with repository ID: {repo_id}")
    
    success = test_wiki_embeddings(repo_id)
    
    if success:
        print("\n✅ TEST PASSED: Embeddings generated successfully for wiki generation using ollama_nomic embeddings.")
    else:
        print("\n❌ TEST FAILED: Could not verify ollama_nomic embeddings for wiki generation.")
        sys.exit(1) 