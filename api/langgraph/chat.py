import os
from datetime import datetime
import traceback
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import re

from api.langgraph.graph import run_rag_pipeline
from api.langgraph.chroma_utils import generate_collection_name, get_chroma_client, check_collection_exists, get_persistent_dir
from api.langgraph.wiki_structure import get_repo_data_dir, normalize_repo_id, get_wiki_data_dir

class Message(BaseModel):
    role: str
    content: str

class ChatHistory(BaseModel):
    repo_id: str
    messages: List[Message] = []
    last_updated: datetime = datetime.now()

# In-memory storage for chat histories
# In a production app, this would be in a database
chat_histories: Dict[str, ChatHistory] = {}

def get_or_create_chat_history(repo_id: str) -> ChatHistory:
    """Get or create a chat history for a repository."""
    if repo_id not in chat_histories:
        chat_histories[repo_id] = ChatHistory(repo_id=repo_id)
    return chat_histories[repo_id]

def add_message_to_history(repo_id: str, message: Message) -> ChatHistory:
    """Add a message to the chat history."""
    history = get_or_create_chat_history(repo_id)
    history.messages.append(message)
    history.last_updated = datetime.now()
    return history

def construct_rag_prompt(query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Construct a prompt for the RAG pipeline that guides the LLM to respond like a principal engineer.
    
    Args:
        query: The user's query
        chat_history: Previous conversation history
        
    Returns:
        A formatted prompt string
    """
    history_str = ""
    if chat_history:
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n\n"
    
    return f"""You are an expert principal engineer and software architect assistant for a codebase. 
Your task is to respond to questions about this repository with detailed, accurate technical information.

CONVERSATION HISTORY:
{history_str}

USER QUERY: {query}

Respond with a detailed, technically precise answer based on the relevant documents provided to you. 
Focus on:
1. Architecture patterns and design principles evident in the code
2. How components interact and relate to each other
3. Best practices and implementation details
4. Potential improvements or architectural considerations

Speak authoritatively as a principal engineer while keeping your answers concise and focused on the technical aspects.
If you don't have enough information from the provided documents, acknowledge that clearly.

YOUR RESPONSE:"""

def resolve_repo_path_for_chat(repo_id: str) -> Dict[str, Any]:
    """
    Resolve a repository ID for chat functionality.
    
    This function tries to find the appropriate repository path and determines
    whether we should use the repository's local files or just the vector store.
    
    Args:
        repo_id: Repository ID which could be a path or a stored ID
        
    Returns:
        Dictionary with resolved path and configuration info
    """
    # Import locally to avoid circular imports
    from api.api import find_wiki_directory
    from api.langgraph.chroma_utils import check_collection_exists, generate_collection_name, get_persistent_dir, get_chroma_client
    from api.langgraph.wiki_structure import normalize_repo_id
    
    # Use ollama_nomic as the embedding provider to match wiki generation
    embedding_provider = "ollama_nomic"
    
    # Normalize the repository ID using our consistent approach
    normalized_repo_id = normalize_repo_id(repo_id)
    print(f"Normalized repository ID: '{normalized_repo_id}' (from '{repo_id}')")
    
    # Create variations for backward compatibility
    repo_id_variations = [
        repo_id,                         # Original ID
        normalized_repo_id,              # Consistently normalized ID
        repo_id.replace('.', '_'),       # For backward compatibility
        repo_id.replace('-', '_'),       # For backward compatibility
        re.sub(r'[^\w]', '_', repo_id),  # Replace all non-word chars with underscore
        re.sub(r'[^a-zA-Z0-9]', '_', repo_id) # Replace all non-alphanumeric with underscore
    ]
    
    # Remove duplicates
    repo_id_variations = list(dict.fromkeys(repo_id_variations))
    
    # Attempt to find a collection using any of the variations
    collection_found = False
    collection_name = None
    persistent_dir = get_persistent_dir()
    client = get_chroma_client(persistent_dir)
    
    for variation in repo_id_variations:
        try:
            # Try to find matching collection using this variation
            temp_collection_name = generate_collection_name(variation)
            if check_collection_exists(client, temp_collection_name):
                collection_found = True
                collection_name = temp_collection_name
                # If we found a collection, update repo_id to the matching variation
                if variation != repo_id:
                    print(f"Found collection using repository ID variation: '{variation}' instead of '{repo_id}'")
                    repo_id = variation
                break
        except Exception as e:
            print(f"Error checking collection for '{variation}': {e}")
    
    if not collection_found:
        error_msg = f"No existing collection found for {repo_id} with {embedding_provider} embeddings. Please generate the wiki first."
        print(f"Error resolving repo path: {error_msg}")
        raise ValueError(error_msg)
    
    # Try to find the repository directory
    try:
        wiki_directory = find_wiki_directory(repo_id)
        print(f"Found wiki directory at: {wiki_directory}")
        
        # If we have a wiki directory, we could use it for additional context
        if wiki_directory:
            result = {
                "repo_identifier": repo_id,
                "use_vectors_only": False,
                "collection_name": collection_name,
                "skip_indexing": True,  # Always use existing collection for chat
                "embedding_provider": embedding_provider,
                "wiki_directory": wiki_directory
            }
            
            # Find repository directory that contains actual code files if it exists
            repo_config = load_repo_config(wiki_directory)
            if repo_config and "repository_directory" in repo_config:
                result["repository_directory"] = repo_config["repository_directory"]
            
            return result
    except Exception as e:
        print(f"Warning: Could not find wiki directory: {e}")
        # Fall back to using just vectors if we can't find the wiki directory
    
    # If we have a collection but no wiki directory, just use the vectors
    return {
        "repo_identifier": repo_id,
        "use_vectors_only": True,
        "collection_name": collection_name,
        "skip_indexing": True,
        "embedding_provider": embedding_provider
    }

def get_chat_response(
    repo_id: str, 
    query: str,
    generator_provider: str = "gemini",
    embedding_provider: str = "ollama_nomic", 
    top_k: int = 10,
    collection_name: str = None
) -> Dict[str, Any]:
    """
    Get a response to a chat query using the RAG pipeline.
    
    Args:
        repo_id: Repository ID
        query: The user's query
        generator_provider: Provider for the generator model (gemini, openai, ollama)
        embedding_provider: Provider for the embedding model (openai, ollama_nomic)
        top_k: Number of documents to retrieve
        collection_name: Optional override for the collection name
        
    Returns:
        Dictionary with answer and metadata
    """
    print(f"Processing chat query for repository: {repo_id}")
    start_time = time.time()
    
    try:
        # Get chat history for context
        history = get_or_create_chat_history(repo_id)
        
        # Add user message to history
        add_message_to_history(repo_id, Message(role="user", content=query))
        
        # Extract previous messages for context (up to 10 most recent messages)
        recent_messages = history.messages[-10:] if len(history.messages) > 10 else history.messages
        chat_history = [{"role": msg.role, "content": msg.content} for msg in recent_messages]
        
        try:
            # Construct the prompt for the LLM
            enhanced_query = construct_rag_prompt(query, chat_history[:-1])  # Exclude the just-added message
            
            # If collection_name is provided, use it directly instead of resolving
            if collection_name:
                repo_config = {
                    "repo_identifier": repo_id,
                    "use_vectors_only": True,
                    "collection_name": collection_name,
                    "skip_indexing": True,
                    "embedding_provider": embedding_provider
                }
                print(f"Using provided collection name: {collection_name}")
                
                # Skip the file system checks since we're using vectors only
                repo_path = None
            else:
                # Resolve the repo_id to a valid path and get config
                repo_config = resolve_repo_path_for_chat(repo_id)
                
                # Determine the correct repository path to use
                # Use the repository_directory from repo_config if available, otherwise fall back to repo_identifier
                if repo_config.get("repository_directory"):
                    repo_path = repo_config["repository_directory"]
                else:
                    repo_path = repo_config["repo_identifier"]
                print(f"Using repository path for RAG: {repo_path}")
            
            # Get collection name and other settings from repo_config
            collection_name = repo_config.get("collection_name")
            skip_indexing = repo_config.get("skip_indexing", True)
            
            # Override embedding provider to match what we resolved
            embedding_provider = repo_config.get("embedding_provider", "ollama_nomic")
            
            # Use the RAG pipeline to generate a response
            response = run_rag_pipeline(
                repo_identifier=None,  # Don't try to use the filesystem
                query=enhanced_query,  # Use the enhanced query with prompt
                generator_provider=generator_provider,
                embedding_provider=embedding_provider,
                top_k=top_k,
                memory={
                    "chat_history": chat_history
                },
                skip_indexing=True,  # Always skip indexing for chat
                collection_name=collection_name
            )
            
            # Process the response to ensure it's in the proper format
            answer = response.get("answer", "")
            
            # Strip any prefixes the LLM might have added like "YOUR RESPONSE:"
            answer = answer.replace("YOUR RESPONSE:", "").strip()
            
            # Add assistant message to history
            add_message_to_history(repo_id, Message(role="assistant", content=answer))
            
            # Enhance response with expert context
            enhanced_answer = enhance_answer_with_expert_context(answer)
            
            # Return the response with metadata
            metadata = response.get("metadata", {})
            
            # Enrich the metadata with timing information
            elapsed_time = time.time() - start_time
            metadata["total_time"] = elapsed_time
            metadata["chat_history_length"] = len(history.messages)
            metadata["vectors_only"] = repo_config.get("use_vectors_only", False)
            metadata["embedding_provider"] = embedding_provider
            
            return {
                "answer": enhanced_answer,
                "metadata": metadata,
                "retrieved_documents": response.get("retrieved_documents", [])
            }
        except Exception as e:
            print(f"Error generating chat response: {e}")
            traceback.print_exc()
            
            # Add error message to chat history
            error_message = f"I encountered an error while trying to answer: {str(e)}"
            add_message_to_history(repo_id, Message(role="assistant", content=error_message))
            
            return {
                "answer": error_message,
                "metadata": {
                    "error": str(e),
                    "error_traceback": traceback.format_exc(),
                    "elapsed_time": time.time() - start_time
                },
                "retrieved_documents": []
            }
    except Exception as e:
        print(f"Unexpected error in get_chat_response: {e}")
        traceback.print_exc()
        return {
            "answer": f"An unexpected error occurred: {str(e)}",
            "metadata": {
                "error": str(e),
                "error_traceback": traceback.format_exc(),
                "elapsed_time": time.time() - start_time
            },
            "retrieved_documents": []
        }

def enhance_answer_with_expert_context(answer: str) -> str:
    """
    Enhance the answer with expert architect or principal engineer context.
    """
    if not answer:
        return "I don't have enough information to answer that question based on the repository content."
    
    # Check if the answer already has a conclusive tone
    has_conclusion = any(phrase in answer.lower() for phrase in [
        "in conclusion", "to summarize", "in summary", "overall", 
        "ultimately", "as a principal engineer", "from an architectural perspective"
    ])
    
    # Add a brief expert conclusion if none exists and the answer is substantial
    if not has_conclusion and len(answer) > 500:
        conclusion = "\n\nFrom an architectural perspective, this approach aligns with modern best practices for maintainable and scalable software design. The codebase demonstrates a thoughtful balance between separation of concerns and pragmatic implementation."
        answer += conclusion
    
    return answer

def load_repo_config(wiki_directory: str) -> dict:
    """
    Load the repository configuration from the wiki directory.
    
    Args:
        wiki_directory: Path to the wiki directory
        
    Returns:
        Repository configuration dictionary
    """
    import json
    import os
    
    config_path = os.path.join(wiki_directory, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading repo config from {config_path}: {e}")
    
    # Check for structure.json as fallback
    structure_path = os.path.join(wiki_directory, "structure.json")
    if os.path.exists(structure_path):
        try:
            with open(structure_path, "r") as f:
                structure = json.load(f)
                if "repository_directory" in structure:
                    return {"repository_directory": structure["repository_directory"]}
        except Exception as e:
            print(f"Error loading structure from {structure_path}: {e}")
    
    return {}

# Test the module if run directly
if __name__ == "__main__":
    test_repo = "/path/to/test/repo"
    test_query = "How is the codebase structured?"
    
    response = get_chat_response(test_repo, test_query)
    print(f"Answer: {response.get('answer')}")
    print(f"Metadata: {response.get('metadata')}") 