"""
Embedding function selection utility for DeepWiki LangGraph pipeline.
"""
import os
import inspect
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

class ChromaOllamaEmbeddingFunction:
    """
    ChromaDB-compatible embedding function for Ollama embeddings.
    
    This class conforms to ChromaDB's expected interface, which requires:
    1. A __call__ method with signature (self, input) where input is a list of strings
    2. A dimensionality property
    """
    def __init__(self, model, base_url):
        from langchain_community.embeddings import OllamaEmbeddings
        self.ollama = OllamaEmbeddings(model=model, base_url=base_url)
        # Try to get a sample embedding to determine dimensionality
        try:
            test_embedding = self.ollama.embed_query("test")
            self._dimensionality = len(test_embedding)
            print(f"Detected Ollama embedding dimensionality: {self._dimensionality}")
        except Exception as e:
            print(f"Warning: Could not determine Ollama embedding dimensionality: {e}")
            self._dimensionality = 768  # Default for nomic-embed-text
    
    def __call__(self, input):
        """
        Generate embeddings for a list of texts.
        
        Args:
            input: A list of strings to embed
            
        Returns:
            A list of embeddings, where each embedding is a list of floats
        """
        # Ensure input is a list
        if not isinstance(input, list):
            input = [input]
        
        # ChromaDB expects a list of embeddings
        return self.ollama.embed_documents(input)
    
    def embed_query(self, text):
        """
        Generate an embedding for a single query string.
        This method is required for compatibility with LangChain's similarity_search.
        
        Args:
            text: The query string to embed
            
        Returns:
            An embedding vector (list of floats)
        """
        return self.ollama.embed_query(text)
    
    @property
    def dimensionality(self) -> int:
        """Return the dimensionality of the embeddings"""
        return self._dimensionality

def get_embedding_function(embedding_provider, api_config=None):
    """
    Get embedding function based on provider.
    
    Args:
        embedding_provider: The embedding provider (openai, ollama_nomic)
        api_config: Optional API configuration object
        
    Returns:
        Embedding function for the specified provider
    """
    if api_config is None:
        # Use simple dictionaries instead of dataclasses to avoid mutable default issues
        default_config = {
            "openai": {
                "model": "text-embedding-3-small",
                "dimensions": 256
            },
            "ollama_nomic": {
                "model": "nomic-embed-text",
                "base_url": "http://localhost:11434"  # Default Ollama URL
            }
        }
    else:
        # Extract config from api_config object if provided
        default_config = {
            "openai": {
                "model": getattr(api_config.embedder, "model", "text-embedding-3-small"),
                "dimensions": getattr(api_config.embedder, "dimensions", 256)
            },
            "ollama_nomic": {
                "model": getattr(api_config.embedder_ollama, "model", "nomic-embed-text"),
                "base_url": getattr(api_config.embedder_ollama, "base_url", "http://localhost:11434")
            }
        }
    
    if embedding_provider == 'openai':
        config = default_config["openai"]
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config["model"], dimensions=config["dimensions"])
    elif embedding_provider == 'ollama_nomic':
        config = default_config["ollama_nomic"]
        return ChromaOllamaEmbeddingFunction(model=config["model"], base_url=config["base_url"])
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}") 