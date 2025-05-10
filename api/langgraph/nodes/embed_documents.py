from api.langgraph.state import RAGState
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import os # For OLLAMA_HOST

def embed_documents_node(state: RAGState) -> RAGState:
    """
    Embeds all chunks using the specified embedding provider (OpenAI or Ollama Nomic).
    Expects:
    - state['chunks']: List of chunks to embed.
    - state['embedding_provider']: 'openai' or 'ollama_nomic'. Defaults to 'openai'.
    Stores embeddings in state['embeddings'] and updates chunk metadata.
    """
    chunks = state.get('chunks', [])
    if not chunks:
        raise ValueError("No chunks found in state for embedding.")

    embedding_provider = state.get('embedding_provider', 'openai') # Default to openai
    
    try:
        from api.langgraph_config import config as api_config
    except ImportError:
        # Fallback minimal config if full config import fails
        from dataclasses import dataclass
        @dataclass
        class MockEmbedderConfig:
            model: str
            dimensions: int = None
            batch_size: int = 1 # Nomic has batch size 1
        @dataclass
        class MockApiConfig:
            embedder: MockEmbedderConfig = MockEmbedderConfig(model="text-embedding-3-small", dimensions=256)
            embedder_ollama: MockEmbedderConfig = MockEmbedderConfig(model="nomic-embed-text", dimensions=768)
        api_config = MockApiConfig()

    embedder = None
    if embedding_provider == 'ollama_nomic':
        print(f"Using Ollama Nomic Embeddings (Model: {api_config.embedder_ollama.model})")
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        embedder = OllamaEmbeddings(
            model=api_config.embedder_ollama.model,
            base_url=ollama_host
        )
    elif embedding_provider == 'openai':
        print(f"Using OpenAI Embeddings (Model: {api_config.embedder.model})")
        embedder = OpenAIEmbeddings(
            model=api_config.embedder.model,
            dimensions=api_config.embedder.dimensions if api_config.embedder.dimensions else None
        )
    else:
        raise ValueError(f"Unsupported embedding_provider: {embedding_provider}. Must be 'openai' or 'ollama_nomic'.")

    texts = [chunk.page_content for chunk in chunks]
    
    # OllamaEmbeddings (especially older versions or certain models) might not handle large batches well.
    # Nomic-embed-text itself might have limitations. Let's process in smaller batches if it's Ollama.
    # OpenAI API has its own batching internally with embed_documents.
    embeddings = []
    if embedding_provider == 'ollama_nomic':
        batch_size = api_config.embedder_ollama.batch_size if hasattr(api_config.embedder_ollama, 'batch_size') else 1
        if batch_size <= 0: batch_size = 1 # Ensure positive batch size
        print(f"Embedding with Ollama in batches of {batch_size}")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedder.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            print(f"Embedded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    else: # OpenAI handles its own batching efficiently
         embeddings = embedder.embed_documents(texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk.metadata['embedding'] = emb

    state['embeddings'] = embeddings
    state['chunks'] = chunks  # Now with embeddings in metadata
    print(f"Successfully generated {len(embeddings)} embeddings using {embedding_provider}.")
    return state

# Usage Example
if __name__ == "__main__":
    from api.langgraph.nodes.split_text import split_text_node
    from api.langgraph.nodes.load_documents import load_documents_node
    # Mock documents for testing
    from langchain_core.documents import Document

    # Test with OpenAI
    state_openai = RAGState()
    # state_openai["repo_identifier"] = "/path/to/repo" # Not needed for this direct test
    state_openai["embedding_provider"] = "openai"
    state_openai["chunks"] = [Document(page_content="This is a test for OpenAI."), Document(page_content="Another OpenAI test.")]
    # state_openai = load_documents_node(state_openai) # Skip for direct test
    # state_openai = split_text_node(state_openai)   # Skip for direct test
    state_openai = embed_documents_node(state_openai)
    print(f"OpenAI Embeddings ({len(state_openai['embeddings'])}): First vector length: {len(state_openai['embeddings'][0]) if state_openai['embeddings'] else 'N/A'}")

    # Test with Ollama Nomic
    # Ensure Ollama is running with nomic-embed-text model pulled: `ollama pull nomic-embed-text`
    state_ollama = RAGState()
    state_ollama["embedding_provider"] = "ollama_nomic"
    state_ollama["chunks"] = [Document(page_content="This is a test for Ollama Nomic."), Document(page_content="Another Ollama test.")]
    state_ollama = embed_documents_node(state_ollama)
    print(f"Ollama Nomic Embeddings ({len(state_ollama['embeddings'])}): First vector length: {len(state_ollama['embeddings'][0]) if state_ollama['embeddings'] else 'N/A'}") 