from api.langgraph.state import RAGState

def generate_node(state: RAGState) -> RAGState:
    """
    Generates an answer using Gemini LLM based on the retrieved documents.
    Expects:
    - state['query']: The user query
    - state['relevant_documents']: The documents retrieved from ChromaDB
    - state['history']: Optional conversation history
    - state['use_ollama']: Optional boolean to use Ollama instead of Gemini/OpenAI
    Returns:
    - state['answer']: The generated answer
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        from api.langgraph_config import config
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class MinimalConfig:
            class Generator:
                model = "gemini-2.5-flash-preview-04-17"
                temperature = 0.7
                top_p = 0.8
            class GeneratorOllama:
                model = "qwen3:1.7b"
                temperature = 0.7
                top_p = 0.8
            generator = Generator()
            generator_ollama = GeneratorOllama()
        config = MinimalConfig()
    query = state.get('query')
    relevant_documents = state.get('relevant_documents', [])
    history = state.get('history', [])
    use_ollama = state.get('use_ollama', False)
    if not query:
        raise ValueError("No 'query' provided in state for generation.")
    MAX_CONTEXT_TOKENS = 900_000
    context_pieces = []
    total_tokens = 0
    truncated = False
    if not relevant_documents:
        context_text = "No specific code context found."
    else:
        for i, doc in enumerate(relevant_documents):
            file_path = doc.metadata.get('file_path', 'Unknown file')
            repo_name = doc.metadata.get('repository', 'Current repository')
            content = doc.page_content
            est_tokens = len(content) // 4
            if total_tokens + est_tokens > MAX_CONTEXT_TOKENS:
                truncated = True
                break
            context_pieces.append(f"Document {i+1} from {file_path} in {repo_name}:\n{content}")
            total_tokens += est_tokens
        context_text = "\n\n".join(context_pieces)
        if truncated:
            print(f"[WARNING] Context truncated to ~{MAX_CONTEXT_TOKENS} tokens. Not all documents included.")
    system_template = """You are an expert code analyst and software documentation assistant.\nAnswer the user's question based on the provided context from the codebase.\nIf the context doesn't contain the information needed, say so clearly rather than making up information.\nFor code-related questions, include relevant code snippets and explain them.\nFor architecture or design questions, provide clear and structured explanations.\nFormat your response using Markdown for readability.\n\nContext from the codebase:\n{context}\n"""
    human_template = "{query}"
    system_message = system_template.format(context=context_text)
    try:
        if use_ollama:
            try:
                from langchain_community.llms.ollama import Ollama
                print(f"Using Ollama model: {config.generator_ollama.model}")
                import os
                ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                llm = Ollama(
                    model=config.generator_ollama.model,
                    base_url=ollama_url,
                    temperature=config.generator_ollama.temperature,
                    top_p=config.generator_ollama.top_p,
                )
            except ImportError:
                print("Warning: langchain_community.llms.ollama not available. Falling back to Gemini.")
                use_ollama = False
            except Exception as e:
                print(f"Error initializing Ollama: {str(e)}. Falling back to Gemini.")
                use_ollama = False
        if not use_ollama:
            print(f"Using Gemini model: {config.generator.model}")
            llm = ChatGoogleGenerativeAI(
                model=config.generator.model,
                temperature=config.generator.temperature,
                top_p=config.generator.top_p,
            )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template),
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"query": query})
        state['answer'] = answer
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(error_msg)
        state['answer'] = f"I couldn't generate a proper answer due to an error: {str(e)}"
        state['error_generate'] = str(e)
    return state

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