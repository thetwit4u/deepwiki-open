import os
import sys
sys.path.append('.')

from api.langgraph.chat import get_chat_response
from api.langgraph.chroma_utils import generate_collection_name, get_chroma_client, get_persistent_dir

def test_chat_with_embeddings():
    """Test that chat functionality uses the correct embedding provider."""
    
    # Check if collections exist
    persistent_dir = get_persistent_dir()
    print(f"ChromaDB directory: {persistent_dir}")
    
    client = get_chroma_client(persistent_dir)
    collection_names = client.list_collections()
    print(f"Total collections: {len(collection_names)}")
    
    if len(collection_names) == 0:
        print("No collections found. Please run generate_embeddings.py first.")
        return
    
    print(f"Available collections: {collection_names}")
    
    # Use the direct collection name without trying to decode it
    collection_name = collection_names[0]
    # For testing, we'll use 'customs.exchange-rate-main'
    # instead of trying to decode from collection name
    repo_id = 'customs.exchange-rate-main'
    
    print(f"Testing chat with repo_id: {repo_id}")
    print(f"Collection name: {collection_name}")
    
    # Try to get a chat response with the collection name override
    try:
        result = get_chat_response(
            repo_id=repo_id, 
            query="What is this repository about?",
            embedding_provider="ollama_nomic",
            generator_provider="gemini",
            collection_name=collection_name
        )
        
        # Check the result
        print("\n=== Chat Response ===")
        print(f"Answer: {result.get('answer', 'No answer')}")
        print("\n=== Metadata ===")
        metadata = result.get('metadata', {})
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        # Verify the embedding provider
        embedding_provider = metadata.get('embedding_provider')
        if embedding_provider == 'ollama_nomic':
            print("\n✅ SUCCESS: Chat is using ollama_nomic embeddings as expected!")
        else:
            print(f"\n❌ ERROR: Chat is using {embedding_provider} embeddings instead of ollama_nomic!")
        
    except Exception as e:
        print(f"Error during chat test: {e}")

if __name__ == "__main__":
    test_chat_with_embeddings() 