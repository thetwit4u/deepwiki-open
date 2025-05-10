"""
Configuration for LangGraph-based RAG pipeline.

This replaces the adalflow-based configuration with LangChain/LangGraph compatible settings.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default values for various configurations
DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 20
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"  # Default Gemini model

# OpenAI config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables.")

# Google Gemini config
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# Default Gemini model from environment variable or fallback to default
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

# Ollama config (for local models)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class EmbedderConfig(BaseModel):
    """Configuration for embedding models."""
    
    provider: str = Field(default="openai", description="Provider for embeddings: 'openai' or 'ollama'")
    model: str = Field(default="text-embedding-3-small", description="Model name for embeddings")
    dimensions: int = Field(default=256, description="Dimensions for OpenAI embeddings")
    batch_size: int = Field(default=500, description="Batch size for processing embeddings")


class RetrieverConfig(BaseModel):
    """Configuration for retrieval settings."""
    
    top_k: int = Field(default=DEFAULT_TOP_K, description="Number of documents to retrieve")
    filter_threshold: Optional[float] = Field(default=None, description="Optional similarity threshold to filter results")


class GeneratorConfig(BaseModel):
    """Configuration for LLM generators."""
    
    provider: str = Field(default="google", description="Provider for LLM: 'google', 'openai', or 'ollama'")
    model: str = Field(default=GEMINI_MODEL, description="Model name for generation")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    top_p: float = Field(default=0.8, description="Top p for generation")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for generation")


class TextSplitterConfig(BaseModel):
    """Configuration for text splitting."""
    
    split_by: str = Field(default="word", description="Unit for splitting: 'word', 'character', 'token'")
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, description="Size of chunks")
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, description="Overlap between chunks")


class LangGraphConfig(BaseModel):
    """Master configuration for LangGraph RAG pipeline."""
    
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    embedder_ollama: Optional[EmbedderConfig] = Field(
        default_factory=lambda: EmbedderConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimensions=768,  # Nomic embed dimensionality
            batch_size=1,  # Ollama doesn't support batching
        )
    )
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    generator_ollama: Optional[GeneratorConfig] = Field(
        default_factory=lambda: GeneratorConfig(
            provider="ollama",
            model="qwen3:1.7b",
            temperature=0.7,
            top_p=0.8,
        )
    )
    text_splitter: TextSplitterConfig = Field(default_factory=TextSplitterConfig)
    persist_dir: str = Field(
        default=os.path.join(os.path.expanduser("~"), ".deepwiki", "chromadb"),
        description="Directory for persisting vector database"
    )
    max_file_size: int = Field(
        default=1024 * 1024,  # 1MB
        description="Maximum file size in bytes for processing"
    )
    update_existing_collections: bool = Field(
        default=False,
        description="Whether to update existing collections (True) or recreate them (False)"
    )


# Create the default configuration
default_config = LangGraphConfig()


def get_config() -> LangGraphConfig:
    """Returns the configuration with environment variable overrides."""
    config = default_config.model_copy()
    
    # Override with environment variables if present
    if os.environ.get("EMBEDDER_MODEL"):
        config.embedder.model = os.environ["EMBEDDER_MODEL"]
    
    if os.environ.get("GENERATOR_MODEL"):
        config.generator.model = os.environ["GENERATOR_MODEL"]
    
    if os.environ.get("GEMINI_MODEL"):
        if config.generator.provider == "google":
            config.generator.model = os.environ["GEMINI_MODEL"]
    
    if os.environ.get("RETRIEVER_TOP_K"):
        config.retriever.top_k = int(os.environ["RETRIEVER_TOP_K"])
    
    if os.environ.get("TEXT_CHUNK_SIZE"):
        config.text_splitter.chunk_size = int(os.environ["TEXT_CHUNK_SIZE"])
    
    if os.environ.get("TEXT_CHUNK_OVERLAP"):
        config.text_splitter.chunk_overlap = int(os.environ["TEXT_CHUNK_OVERLAP"])
    
    if os.environ.get("PERSIST_DIR"):
        config.persist_dir = os.environ["PERSIST_DIR"]
    
    return config


# Export a singleton instance of the configuration
config = get_config() 