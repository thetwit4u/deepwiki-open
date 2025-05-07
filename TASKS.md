# DeepWiki Refactoring & Integration Tasks

This document tracks the progress and requirements for refactoring DeepWiki to use the new LangGraph-based RAG backend, and for fully integrating it with a modern, modular frontend.

---

## Phase 1: Backend Refactoring & Core Features

### Step 1.1: LangGraph RAG Pipeline Foundation
- [x] Create `api/langgraph_rag.py` with modular node-based pipeline
- [x] Implement document loading (Git repo & local directory)
    - [x] CLI test for document loading
- [x] Implement text splitting with configurable chunking
    - [x] CLI test for text splitting
- [x] Implement embedding node (OpenAI)
    - [x] CLI test for embedding node
- [x] Implement vector storage node (ChromaDB)
    - [x] CLI test for vector storage
- [x] Implement retrieval node
    - [x] CLI test for retrieval
- [x] Implement answer generation node (Gemini/OpenAI)
    - [x] CLI test for full pipeline

---

## Phase 2: Incremental Frontend + Backend Feature Development

For each major feature, implement both the backend API and the frontend UI in tandem. Test end-to-end before moving to the next feature. This ensures rapid feedback and modular, testable progress.

### Step 2.1: Project/Repo Selection
- [ ] Backend: API endpoint to list/select available repositories
- [ ] Frontend: UI for selecting a repository (Next.js, shadcn/ui, Tailwind)
- [ ] Test: End-to-end selection and state propagation

### Step 2.2: Wiki Structure Overview
- [ ] Backend: API to return wiki structure (sections, hierarchy)
- [ ] Frontend: UI to display wiki structure (sidebar/tree view)
- [ ] Test: Structure loads and displays correctly

### Step 2.3: Section Rendering (Content Pages)
- [ ] Backend: API to return section content (with prompts for consistency)
- [ ] Frontend: UI to render section content (markdown/code/mermaid)
- [ ] Test: Section loads and renders as expected

### Step 2.4: Technical Summary Section
- [ ] Backend: API to generate technical summary (languages, AWS, DBs, etc.)
- [ ] Frontend: UI to display technical summary
- [ ] Test: Summary is accurate and well-presented

### Step 2.5: Mermaid Diagrams
- [ ] Backend: API to generate mermaid diagrams for relevant sections
- [ ] Frontend: UI to render mermaid diagrams (live preview)
- [ ] Test: Diagrams render and update correctly

### Step 2.6: Search & Navigation
- [ ] Backend: API for full-text search across wiki
- [ ] Frontend: UI for search and navigation
- [ ] Test: Search results are relevant and navigation is smooth

---

## Phase 3: Integration, Testing, and Polish
- [ ] End-to-end integration tests for all features
- [ ] Accessibility and UX review
- [ ] Performance optimization (backend and frontend)
- [ ] Documentation and usage examples

---

## Phase 4: UI Refactor & AWS-Ready Deployment
- [ ] Modularize UI components (Next.js app directory, shadcn/ui, Tailwind)
- [ ] Ensure clean black and white theme
- [ ] Prepare for easy AWS deployment (Vercel/Amplify/EC2)
- [ ] Provide deployment scripts and instructions

### Proposed UI Structure (Next.js)
```
/src
  /app
    /[repo]         # Dynamic repo pages
      /[section]    # Dynamic section pages
    /api            # API routes (Next.js)
  /components       # Modular UI components (sidebar, section, summary, diagram, etc.)
  /lib              # Shared utilities
  /styles           # Tailwind config, global styles
```
- Use shadcn/ui for all UI primitives
- Use Tailwind for layout and theming
- Keep all features modular and testable

---

## Notes
- Ollama support is deferred (nice to have)
- Prompts for wiki structure and sections must be reviewed for consistency and quality
- Each section prompt should specify requirements and mermaid diagram expectations
- Every wiki must include a technical summary section (languages, AWS, DBs, etc.)

## Dependency Graph

```mermaid
graph TD
    A[Update requirements.txt] --> B[Create LangGraph structure]
    B --> C[Create configuration system]
    C --> D[Implement RAG orchestration]
    D --> E[Test individual nodes]
    E --> F[Test full graph]
    
    F --> G[Implement ChromaDB]
    G --> H[Define metadata schema]
    H --> I[Integrate with LangGraph]
    I --> J[Test storage and retrieval]
    
    J --> K[Enhance metadata schema]
    K --> L[Implement query filtering]
    L --> M[Update API request model]
    M --> N[Create merged results handling]
    N --> O[Test multi-repo querying]
```

## Next Steps After Phase 1

Once Phase 1 is complete, we'll proceed to:
- Phase 2: Frontend & API Adjustments for Local Paths
- Phase 3: Testing and Refinement 