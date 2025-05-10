from api.langgraph.state import RAGState
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def split_text_node(state: RAGState) -> RAGState:
    """
    Efficiently splits code and markdown/text files into chunks.
    Stores resulting chunks in state['chunks'].
    """
    documents = state.get("documents", [])
    if not documents:
        raise ValueError("No documents found in state for splitting.")
    code_exts = {'.py', '.js', '.ts', '.tsx', '.java', '.go', '.cpp', '.c', '.h', '.cs', '.rb', '.php', '.sh'}
    markdown_exts = {'.md', '.markdown', '.rst'}
    text_exts = {'.txt', '.json', '.yaml', '.yml'}
    all_chunks = []
    for doc in documents:
        ext = os.path.splitext(doc.metadata.get('file_path', ''))[1].lower()
        text = doc.page_content if hasattr(doc, 'page_content') else doc.text
        meta = dict(doc.metadata)
        if ext in code_exts:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=60,
                chunk_overlap=10,
                separators=["\nclass ", "\ndef ", "\nfunction ", "\n# ", "\n", " ", ""]
            )
        elif ext in markdown_exts:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n## ", "\n# ", "\n", " ", ""]
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n", " ", ""]
            )
        splits = splitter.split_text(text)
        for i, chunk in enumerate(splits):
            chunk_meta = meta.copy()
            chunk_meta['chunk_index'] = i
            chunk_meta['chunk_total'] = len(splits)
            chunk_meta['chunk_size'] = len(chunk)
            chunk_meta['original_ext'] = ext
            all_chunks.append(Document(page_content=chunk, metadata=chunk_meta))
    state['chunks'] = all_chunks
    return state

# Usage Example
if __name__ == "__main__":
    from api.langgraph.nodes.load_documents import load_documents_node
    state = RAGState()
    state["repo_identifier"] = "/path/to/repo"
    state = load_documents_node(state)
    state = split_text_node(state)
    print(len(state["chunks"])) 