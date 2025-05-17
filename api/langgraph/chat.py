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

def resolve_repo_path_for_chat(repo_id: str, collection_name: str = None) -> Dict[str, Any]:
    """
    Resolve a repository ID for chat functionality.
    
    This function tries to find the appropriate repository path and determines
    whether we should use the repository's local files or just the vector store.
    
    Args:
        repo_id: Repository ID which could be a path or a stored ID
        collection_name: Optional override for ChromaDB collection name
        
    Returns:
        Dictionary with resolved path and configuration info
    """
    # Import locally to avoid circular imports
    from api.api import find_wiki_directory
    from api.langgraph.chroma_utils import check_collection_exists, generate_collection_name, get_persistent_dir, get_chroma_client
    from api.langgraph.wiki_structure import normalize_repo_id
    
    embedding_provider = "ollama_nomic"
    normalized_repo_id = normalize_repo_id(repo_id)
    print(f"Normalized repository ID: '{normalized_repo_id}' (from '{repo_id}')")
    
    # If collection_name is provided, skip the collection search
    if collection_name:
        persistent_dir = get_persistent_dir()
        client = get_chroma_client(persistent_dir)
        if check_collection_exists(client, collection_name):
            print(f"Using provided collection name: {collection_name}")
        else:
            print(f"Warning: Provided collection '{collection_name}' doesn't exist")
    else:
        # Always list all collections and filter for those starting with normalized_repo_id
        persistent_dir = get_persistent_dir()
        client = get_chroma_client(persistent_dir)
        all_collections = client.list_collections()
        # Convert to string if needed
        all_collection_names = [str(c) for c in all_collections]
        # Find any collection whose name starts with 'local_{normalized_repo_id}_'
        prefix = f"local_{normalized_repo_id}_"
        matching = [name for name in all_collection_names if name.startswith(prefix)]
        if matching:
            collection_name = matching[0]
            print(f"Found collection by prefix match: {collection_name}")
        else:
            # Fallback to generated collection name
            temp_collection_name = generate_collection_name(normalized_repo_id)
            if check_collection_exists(client, temp_collection_name):
                collection_name = temp_collection_name
                print(f"Found collection by generated name: {collection_name}")
            else:
                error_msg = f"No existing collection found for {repo_id} with {embedding_provider} embeddings. Please generate the wiki first."
                print(f"Error resolving repo path: {error_msg}")
                raise ValueError(error_msg)
    
    # Try to find the repository directory
    try:
        wiki_dir_result = find_wiki_directory(repo_id)
        # Handle both string and tuple returns (path, is_legacy) from find_wiki_directory
        if isinstance(wiki_dir_result, tuple):
            wiki_directory = wiki_dir_result[0]
        else:
            wiki_directory = wiki_dir_result
        
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

def get_chat_response(repo_id, query, generator_provider="gemini", embedding_provider="ollama_nomic", top_k=10, collection_name=None):
    """
    Generate a response to a user query about a repository using a conversational RAG approach.
    
    This function:
    1. Gets the conversation history for the repository
    2. Adds the new query to the history
    3. Retrieves relevant documents from the repository
    4. Generates a response based on the retrieved documents and conversation history
    5. Adds the response to the conversation history
    6. Returns the response
    
    Args:
        repo_id: The repository ID or URL
        query: The user query
        generator_provider: The generator model provider (gemini, openai, ollama)
        embedding_provider: The embedding model provider (openai, ollama_nomic)
        top_k: The number of documents to retrieve
        collection_name: Optional override for the ChromaDB collection name
    
    Returns:
        A dictionary containing the response text and metadata
    """
    print(f"[DEBUG get_chat_response] Processing query for repo '{repo_id}' with query '{query}'")
    print(f"[DEBUG get_chat_response] Using model: {generator_provider}, embeddings: {embedding_provider}")
    if collection_name:
        print(f"[DEBUG get_chat_response] Using provided collection name: {collection_name}")
    
    start_time = time.time()
    
    try:
        # 1. Get or create chat history for this repository
        history = get_or_create_chat_history(repo_id)
        
        # 2. Add the new query to history
        add_message_to_history(repo_id, Message(role="user", content=query))
        
        try:
            # 3. Get repository configuration from chat_repositories.json
            repo_config = resolve_repo_path_for_chat(repo_id, collection_name)
            
            # Resolve the repository path for this chat
            if repo_config.get("repository_directory"):
                repo_path = repo_config["repository_directory"]
            else:
                repo_path = repo_config["repo_identifier"]
            
            # If no collection_name was provided, attempt to find the correct collection
            if not collection_name:
                # First try to get it from repo config
                collection_name = repo_config.get("collection_name")
                
                # If not in repo config, dynamically look it up using the lookup_collection logic
                if not collection_name:
                    from api.langgraph.chroma_utils import get_chroma_client, generate_collection_name, check_collection_exists
                    
                    # Normalize the repository ID for consistent handling
                    from api.langgraph.wiki_structure import normalize_repo_id
                    normalized_repo_id = normalize_repo_id(repo_id)
                    
                    # Create variations of the repo ID to try matching with collections
                    import re
                    repo_id_variations = [
                        repo_id,                           # Original ID
                        normalized_repo_id,                # Consistently normalized ID
                        repo_id.replace('.', '_'),         # For backward compatibility
                        repo_id.replace('-', '_'),         # For backward compatibility
                        re.sub(r'[^\w]', '_', repo_id),    # Replace all non-word chars with underscore
                        re.sub(r'[^a-zA-Z0-9]', '_', repo_id) # Replace all non-alphanumeric with underscore
                    ]
                    # Remove duplicates
                    repo_id_variations = list(dict.fromkeys(repo_id_variations))
                    
                    # Get ChromaDB client
                    from api.langgraph.chroma_utils import get_persistent_dir
                    client = get_chroma_client(get_persistent_dir())
                    
                    # Try each variation to find a matching collection
                    for variation in repo_id_variations:
                        try:
                            temp_collection_name = generate_collection_name(variation)
                            if check_collection_exists(client, temp_collection_name):
                                collection_name = temp_collection_name
                                print(f"[DEBUG get_chat_response] Found collection using repository ID variation: '{variation}' -> '{collection_name}'")
                                break
                        except Exception as e:
                            print(f"[DEBUG get_chat_response] Error checking collection for '{variation}': {e}")
                    
                    # If still no match, try the custom collection naming pattern for problematic repos
                    if not collection_name and repo_id == "customs_exchange_rate_main":
                        collection_name = "local_customs_exchange_rate_main_9cfa74b61a"
                        print(f"[DEBUG get_chat_response] Using special hardcoded collection for known problematic repository: {collection_name}")
            
            print(f"[DEBUG get_chat_response] Using collection name: {collection_name}")
            
            # 4. Generate a response using the RAG pipeline
            response = run_rag_pipeline(
                repo_identifier=repo_path,
                query=query,
                generator_provider=generator_provider,
                embedding_provider=embedding_provider,
                top_k=top_k,
                collection_name=collection_name,
                skip_indexing=True
            )
            
            # 5. Add the response to history
            add_message_to_history(repo_id, Message(role="assistant", content=response["answer"]))
            
            # Return the response with metadata
            metadata = response.get("metadata", {})
            
            # Enrich the metadata with timing information
            elapsed_time = time.time() - start_time
            metadata["total_time"] = elapsed_time
            metadata["chat_history_length"] = len(history.messages)
            metadata["vectors_only"] = repo_config.get("use_vectors_only", False)
            metadata["embedding_provider"] = embedding_provider
            metadata["collection_name"] = collection_name
            
            # Convert LangChain Document objects to dictionaries to ensure they can be serialized
            retrieved_documents = response.get("retrieved_documents", [])
            serializable_documents = []
            
            for doc in retrieved_documents:
                # Check if it's already a dict
                if isinstance(doc, dict):
                    serializable_documents.append(doc)
                # Check if it's a LangChain Document object
                elif hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    # Convert Document to a serializable dictionary
                    serializable_documents.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })
                else:
                    # For any other type, try to convert to dict or use string representation
                    try:
                        serializable_documents.append(dict(doc))
                    except (TypeError, ValueError):
                        serializable_documents.append({"content": str(doc)})
            
            return {
                "answer": response["answer"],
                "metadata": metadata,
                "retrieved_documents": serializable_documents
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
                    "elapsed_time": time.time() - start_time,
                    "collection_name": collection_name  # Include the collection name in error response
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