"""
Generation node for the RAG pipeline.

Inputs from state:
- state['query']: User query
- state['relevant_documents']: The documents retrieved from ChromaDB

Outputs added to state:
- state['answer']: The generated answer
"""
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_core.documents import Document

def format_documents(docs: List[Document]) -> str:
    """Format a list of documents into a string."""
    return "\n\n".join([
        f"DOCUMENT [{i+1}] (Source: {doc.metadata.get('source', 'Unknown')})\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

def format_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> str:
    """Format chat history into a string for the prompt."""
    if not chat_history:
        return ""
    
    formatted_history = []
    for message in chat_history:
        role = "User" if message["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {message['content']}")
    
    return "\n\n".join(formatted_history)

def generate_prompt(query: str, docs: List[Document], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generate a prompt for the LLM.
    
    Args:
        query: User query (may already contain a formatted prompt)
        docs: Retrieved documents
        chat_history: Optional chat history
        
    Returns:
        Formatted prompt string
    """
    # Check if the query already contains a detailed prompt (from the chat module)
    if "USER QUERY:" in query and "CONVERSATION HISTORY:" in query:
        # The query already contains a formatted prompt, just append the documents
        documents_str = format_documents(docs)
        return f"{query}\n\nRELEVANT DOCUMENTS:\n{documents_str}"
    
    # Otherwise, create a prompt from scratch
    formatted_docs = format_documents(docs)
    formatted_history = format_chat_history(chat_history) if chat_history else ""
    
    # Create different prompts based on whether we have chat history
    if formatted_history:
        return f"""You are an expert software developer and technical assistant.
        
CHAT HISTORY:
{formatted_history}

USER QUERY: {query}

Use the following documents to provide a detailed, accurate answer to the query.

RELEVANT DOCUMENTS:
{formatted_docs}

Based on the documents provided, respond to the user's query with accurate, technical information.
If the documents don't contain sufficient information to answer the query, acknowledge this limitation.
"""
    else:
        return f"""You are an expert software developer and technical assistant.

USER QUERY: {query}

Use the following documents to provide a detailed, accurate answer to the query.

RELEVANT DOCUMENTS:
{formatted_docs}

Based on the documents provided, respond to the user's query with accurate, technical information.
If the documents don't contain sufficient information to answer the query, acknowledge this limitation.
"""

def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an answer based on the query and relevant documents."""
    try:
        query = state.get("query", "")
        relevant_docs = state.get("relevant_documents", [])
        memory = state.get("memory", {})
        chat_history = memory.get("chat_history", []) if memory else []
        use_ollama = state.get("use_ollama", False)
        
        if not relevant_docs:
            state["answer"] = "No relevant documents were found to answer your query."
            return state
        
        # Create a prompt for the LLM
        prompt = generate_prompt(query, relevant_docs, chat_history)
        
        # Use either Google's Gemini or OpenAI for generation
        if use_ollama:
            answer = generate_with_ollama(prompt)
        else:
            answer = generate_with_gemini(prompt)
        
        state["answer"] = answer
        return state
        
    except Exception as e:
        print(f"Error in generate_node: {e}")
        traceback.print_exc()
        state["answer"] = f"Error generating answer: {str(e)}"
        state["error_generate"] = str(e)
        return state

def generate_with_gemini(prompt: str) -> str:
    """Generate an answer using Google's Gemini model."""
    import google.generativeai as genai
    import os
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY environment variable not set."
    
    genai.configure(api_key=api_key)
    
    try:
        # Import here to access any config that may have been set up in the main script
        from api.langgraph_config import config
        model_name = config.generator.model
        temperature = config.generator.temperature
        top_p = config.generator.top_p
        top_k = getattr(config.generator, "top_k", 32)
    except ImportError:
        # Default config if langgraph_config cannot be imported
        model_name = "gemini-2.5-flash-preview-04-17"
        temperature = 0.7
        top_p = 0.8
        top_k = 40
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": 2048,
        }
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error with Gemini generation: {e}")
        # Try again with a simpler prompt
        try:
            fallback_prompt = f"Summarize the following information:\n\n{prompt}"
            response = model.generate_content(fallback_prompt)
            return response.text + "\n\nNote: This response was generated with a fallback prompt due to an error."
        except Exception as fallback_error:
            print(f"Error with fallback Gemini generation: {fallback_error}")
            return f"Error generating content with Gemini: {str(e)}"

def generate_with_ollama(prompt: str) -> str:
    """Generate an answer using Ollama."""
    import requests
    
    try:
        # Import here to access any config that may have been set up in the main script
        from api.langgraph_config import config
        model_name = config.generator_ollama.model
        temperature = config.generator_ollama.temperature
        top_p = config.generator_ollama.top_p
        top_k = getattr(config.generator_ollama, "top_k", 40)
    except ImportError:
        # Default config if langgraph_config cannot be imported
        model_name = "qwen3:1.7b"
        temperature = 0.7
        top_p = 0.8
        top_k = 40
    
    api_url = "http://localhost:11434/api/generate"
    
    try:
        response = requests.post(
            api_url,
            json={
                "model": model_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": 1024,
                "stream": False
            },
            timeout=60  # 60 second timeout
        )
        response.raise_for_status()
        return response.json().get("response", "No response received from Ollama")
    except requests.exceptions.RequestException as e:
        print(f"Error with Ollama generation: {e}")
        return f"Error generating content with Ollama: {str(e)}"

# Usage Example
if __name__ == "__main__":
    from api.langgraph.nodes.retrieve import retrieve_node
    from api.langgraph.nodes.store_vectors import store_vectors_node
    from api.langgraph.nodes.embed_documents import embed_documents_node
    from api.langgraph.nodes.split_text import split_text_node
    from api.langgraph.nodes.load_documents import load_documents_node
    state = RAGState()
    state["repo_identifier"] = "/path/to/repo"
    state["query"] = "What does this repo do?"
    state = load_documents_node(state)
    state = split_text_node(state)
    state = embed_documents_node(state)
    state = store_vectors_node(state)
    state = retrieve_node(state)
    state = generate_node(state)
    print(state.get("answer")) 