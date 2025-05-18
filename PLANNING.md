# DeepWiki Refactoring Plan

This document outlines the plan to refactor the DeepWiki application based on the requirements to replace `adalflow` with `langgraph`, enhance vector database persistence, support local repository paths, and enable efficient multi-repository querying.

## 1. Overall Goals

*   **Modernize Agentic Workflow**: Replace the existing `adalflow`-based RAG pipeline with `langgraph` for more flexible and extensible AI agent development.
*   **Robust Local Data Management**: Ensure vector database an_data are persisted locally in a standard, manageable format, independent of `adalflow`.
*   **Enhanced Input Flexibility**: Allow users to specify local file system paths as repositories, in addition to Git URLs.
*   **Advanced Querying Capabilities**: Structure the vector database to support efficient querying of information related to a single repository, multiple specified repositories, or all indexed repositories.

## 2. Current State Analysis Summary

*   **`adalflow` Integration**: Deeply integrated for RAG pipeline components (embedding, generation, retrieval), data processing, and local database management (`LocalDB` using FAISS and pickle files). Key files: `api/rag.py`, `api/data_pipeline.py`, `api/ollama_patch.py`, `api/config.py`.
*   **VectorDB Persistence**: Uses `adalflow.core.db.LocalDB`, which saves a FAISS index and metadata as a single pickle file per repository (`~/.adalflow/databases/{repo_name}.pkl`).
*   **Repository Input**:
    *   Frontend (`src/app/page.tsx`): Accepts GitHub, GitLab, Bitbucket URLs, and `owner/repo` format. No local path support.
    *   Backend (`api/simple_chat.py`, `api/data_pipeline.py`): Expects `repo_url` and clones Git repositories.
*   **Multi-Repo Querying**: Not explicitly supported. Each repository has its own isolated database file.

## 3. Proposed Refactoring Steps

### Phase 1: Core Backend Refactoring (LangGraph & VectorDB)

#### Step 1.1: Introduce LangGraph and Basic RAG Pipeline (Impact: Large)

*   **Objective**: Replace `adalflow`'s RAG mechanics with a `langgraph`-based solution.
*   **Tasks**:
    *   Add `langgraph` and necessary dependencies (e.g., `langchain-community` for vector stores if needed) to `api/requirements.txt`.
    *   Define `langgraph` graph(s) to replicate the current RAG flow:
        *   Node for loading documents (from URLs or local paths).
        *   Node for text splitting.
        *   Node for embedding generation (supporting OpenAI and Ollama).
        *   Node for vector storage and retrieval.
        *   Node for query understanding/rewriting (optional, for future enhancement).
        *   Node for language model interaction (generation).
        *   Node for managing conversation history.
    *   Refactor `api/rag.py` to use the new `langgraph` RAG pipeline.
    *   Update model client configurations in `api/config.py` to be compatible with `langchain` (or new `langgraph` paradigms).
    *   Remove `adalflow` dependencies progressively as components are replaced.
    *   The `api/ollama_patch.py` might need to be adapted or its logic integrated into the new embedding node if `langchain`'s Ollama embedder has similar batching limitations.

#### Step 1.2: Implement Local Vector Database with ChromaDB (Impact: Medium)

*   **Objective**: Replace `adalflow.LocalDB` with ChromaDB for robust, queryable local persistence.
*   **Tasks**:
    *   Add `chromadb` to `api/requirements.txt`.
    *   Modify `api/data_pipeline.py` (or create a new data management module):
        *   Replace `DatabaseManager`'s `LocalDB` usage with ChromaDB client.
        *   Implement logic to create/load a persistent ChromaDB collection.
        *   Initially, continue with one collection per repository. The collection name can be derived from the repository identifier (URL or sanitized local path).
        *   Store document chunks and their embeddings in ChromaDB.
        *   Ensure metadata (e.g., `file_path`, `type`, `is_code`, `repository_id`) is stored alongside vectors in ChromaDB. The `repository_id` will be crucial for multi-repo querying.
    *   Update the `langgraph` retrieval node to query ChromaDB.
    *   Define a clear schema for metadata to be stored with vectors.

#### Step 1.3: Modify Data Storage for Multi-Repository Querying (Impact: Medium)

*   **Objective**: Structure ChromaDB to allow querying across multiple repositories.
*   **Tasks**:
    *   Ensure each document/vector stored in ChromaDB includes a `repository_id` in its metadata. This ID should be unique for each repository (e.g., sanitized URL or a hash of the local path).
    *   When querying:
        *   For a single repository context: Filter ChromaDB queries by the active `repository_id`.
        *   For multiple/all repositories:
            *   The API will need a way to specify if a query is global or targets specific multiple repositories.
            *   Queries to ChromaDB will either omit the `repository_id` filter (for all repos) or use an `$in` style filter if querying a subset of repositories.
            *   The RAG pipeline will need to handle and potentially merge/rank results from different repositories.
    *   This step might involve changes to the chat API request model to specify the scope of a query (current repo, all repos, list of repos).

### Phase 2: Frontend & API Adjustments for Local Paths

#### Step 2.1: Update Frontend for Local Path Input (Impact: Medium)

*   **Objective**: Allow users to input local directory paths in the UI.
*   **Tasks (`src/app/page.tsx`):**
    *   Modify the `parseRepositoryInput` function or add new logic to detect if the input string is a local file path (e.g., starts with `/`, `\`, `C:\`, or is not a valid URL).
    *   Update the input field's placeholder and any validation messages to inform users about local path support.
    *   Change how the form submission data is prepared. Instead of `owner` and `repo` for the route, it might need to pass the raw local path (or a sanitized/encoded version) as a query parameter or use a different route for local repositories.
    *   Consider adding a "Type" selector (URL/Local Path) to make the input method explicit for the user.

#### Step 2.2: Update Backend API to Handle Local Paths (Impact: Medium)

*   **Objective**: Enable the backend to process local repository paths.
*   **Tasks**:
    *   **API Model (`api/simple_chat.py`)**:
        *   Modify `ChatCompletionRequest` to accept a local path. This could be a new field `repo_path: Optional[str]` or by making `repo_url` more generic (e.g., `repo_identifier: str`).
    *   **Endpoint Logic (`api/simple_chat.py`)**:
        *   Update `chat_completions_stream` to distinguish between a URL and a local path in the request.
        *   Pass the correct identifier (URL or local path) to the RAG/data pipeline.
    *   **Data Pipeline (`api/data_pipeline.py` or new module)**:
        *   The `DatabaseManager` (or its replacement) must handle `repo_url_or_path`:
            *   If it's a URL, existing cloning logic applies (though this might be simplified if `langchain` offers Git loaders).
            *   If it's a local path, bypass cloning and use the path directly to read files (e.g., using `langchain_community.document_loaders.FileSystemLoader`).
        *   Develop a consistent method to generate a `repository_id` for local paths (e.g., hash of the absolute path, or a sanitized version of the directory name) to be used for ChromaDB collection naming and metadata.

### Phase 3: Testing and Refinement

#### Step 3.1: Incremental Testing

*   **Objective**: Ensure each refactored component works as expected.
*   **Tasks**:
    *   Test the new `langgraph`-based RAG pipeline with a single repository.
    *   Verify ChromaDB persistence: stop/start the application and ensure data is reloaded.
    *   Test local path input from UI to backend processing.
    *   Test querying a single repository via the chat interface.
    *   Test multi-repository querying (if a mechanism is added to the UI/API to trigger this).

#### Step 3.2: Documentation and Cleanup

*   **Objective**: Update documentation and remove old code.
*   **Tasks**:
    *   Update `README.md` with new setup instructions, dependencies, and features.
    *   Remove all `adalflow` related code and `~/.adalflow` directory references (once data is migrated or the new system is stable).
    *   Document the new vector database structure and how to query it.
    *   Clean up any temporary or unused code.

## 4. Potential Challenges & Considerations

*   **Complexity of `adalflow` Replacement**: `adalflow` provides many abstractions. Replicating all functionalities and ensuring feature parity with `langgraph` will require careful design.
*   **Data Migration**: If existing `.pkl` databases need to be migrated to ChromaDB, a script or process will be necessary. For a fresh start, this might not be needed.
*   **Performance**: Embedding and indexing large repositories can be time-consuming. Performance of ChromaDB with many repositories/vectors needs to be monitored.
*   **Error Handling**: Robust error handling for local path access (permissions, non-existent paths) and database operations is crucial.
*   **UI for Multi-Repo Queries**: Designing an intuitive UI for users to specify whether they want to query the current repo, all repos, or a selection of repos will require thought.
*   **Data Migration**: Not required, as confirmed. We will start fresh with ChromaDB.
*   **Performance**: Embedding and indexing large repositories can be time-consuming. Performance of ChromaDB with many repositories/vectors needs to be monitored.

## 5. Impact Levels Summary

*   **Replace `adalflow` with `langgraph`**: Large
*   **VectorDB Persistence (ChromaDB)**: Medium (linked to `adalflow` replacement)
*   **Support Local Repository Paths**: Medium (Frontend + Backend)
*   **Multi-Repo Querying**: Medium to Large (DB design, API changes, RAG pipeline adjustments)

This plan provides a structured approach to the refactoring. Each phase and step should be implemented iteratively, with testing at each stage.

## 6. Roadmap & Future Enhancements

- Integrate PydanticAI for advanced configuration, validation, and dynamic pipeline construction (planned)
- Multi-repository querying UI and API
- Full-text search across all indexed content
- Advanced user settings and export features
- Further optimize performance and accessibility

> **Note:**
> The current backend is fully LangGraph-based. All planning and future enhancements are for the new architecture. The legacy adalflow system is no longer relevant. 