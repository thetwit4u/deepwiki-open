from api.langgraph.state import RAGState, ConversationMemory

def memory_node(state: RAGState) -> RAGState:
    """
    Manages conversation memory by storing the current query and answer.
    Expects:
    - state['query']: The user query
    - state['answer']: The generated answer (optional if an error occurred)
    - state['memory']: Optional existing ConversationMemory object
    Returns:
    - state['memory']: Updated conversation memory
    """
    query = state.get('query')
    answer = state.get('answer')
    if not query:
        error_msg = "Missing 'query' in state for memory."
        print(error_msg)
        state['error_memory'] = error_msg
        return state
    if not answer:
        error_message = state.get('error_generate', "An unknown error occurred")
        answer = f"I'm sorry, I couldn't generate a proper response. Error: {error_message}"
        state['answer'] = answer
        print(f"Created fallback answer for memory: {answer[:50]}...")
    memory = state.get('memory')
    if not memory or not isinstance(memory, ConversationMemory):
        memory = ConversationMemory()
        print("Created new ConversationMemory instance")
    try:
        memory.add_dialog_turn(user_content=query, assistant_content=answer)
        print(f"Added dialog turn to memory with query: {query[:30]}... and answer: {answer[:30]}...")
    except Exception as e:
        error_msg = f"Error adding dialog turn to memory: {str(e)}"
        print(error_msg)
        state['error_memory'] = error_msg
        return state
    state['memory'] = memory
    print("Successfully updated conversation memory")
    return state

# Usage Example
if __name__ == "__main__":
    from api.langgraph.nodes.generate import generate_node
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
    state = memory_node(state)
    print(state.get("memory").to_dict()) 