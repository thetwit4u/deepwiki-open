"""
Embedding function selection utility for DeepWiki LangGraph pipeline.
"""
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function(embedding_provider: str, api_config):
    """
    Returns the appropriate embedding function instance based on the provider string and config.
    """
    if embedding_provider == 'ollama_nomic':
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaEmbeddings(
            model=api_config.embedder_ollama.model,
            base_url=ollama_host
        )
    elif embedding_provider == 'openai':
        return OpenAIEmbeddings(
            model=api_config.embedder.model,
            dimensions=api_config.embedder.dimensions if api_config.embedder.dimensions else None
        )
    raise ValueError(f"Unsupported embedding_provider: {embedding_provider}") 