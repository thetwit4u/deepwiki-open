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

def get_embedding_function(embedding_provider: str, api_config):
    """
    Returns the appropriate embedding function instance based on the provider string and config.
    """
    if embedding_provider == 'ollama_nomic':
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return ChromaOllamaEmbeddingFunction(
            model=api_config.embedder_ollama.model,
            base_url=ollama_host
        )
    elif embedding_provider == 'openai':
        return OpenAIEmbeddings(
            model=api_config.embedder.model,
            dimensions=api_config.embedder.dimensions if hasattr(api_config.embedder, 'dimensions') else None
        )
    raise ValueError(f"Unsupported embedding_provider: {embedding_provider}") 