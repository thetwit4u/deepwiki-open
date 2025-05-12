"""
Wiki structure and content generation utilities for DeepWiki.
This module was refactored from the old monolithic langgraph_rag.py.
"""
import os as _os
import json as _json
import hashlib
from datetime import datetime as _datetime
from typing import List
import re
import yaml

def normalize_repo_id(repo_identifier: str) -> str:
    """Create a safe directory name for the repo."""
    if repo_identifier.startswith("http://") or repo_identifier.startswith("https://"):
        slug = repo_identifier.rstrip("/").split("/")[-1].replace(".git", "")
        owner = repo_identifier.rstrip("/").split("/")[-2]
        base = f"{owner}_{slug}"
    else:
        base = _os.path.basename(_os.path.abspath(repo_identifier))
    
    # Remove any unusual characters that might cause issues
    base = re.sub(r'[^\w\-_\.]', '_', base)
    
    return base

def get_wiki_data_dir() -> str:
    """
    Get the base directory for all wiki data.
    - If the environment variable DEEPWIKI_DATA_DIR is set, use that.
    - Otherwise, use <project-root>/wiki-data, where project root is the parent of the api/ directory.
    """
    env_dir = _os.environ.get("DEEPWIKI_DATA_DIR")
    if env_dir:
        base_dir = env_dir
    else:
        # Compute project root as parent of the directory containing this file's parent (i.e., api/)
        this_dir = _os.path.dirname(__file__)
        api_dir = _os.path.dirname(this_dir)
        project_root = _os.path.dirname(api_dir)
        base_dir = _os.path.join(project_root, "wiki-data")
    _os.makedirs(base_dir, exist_ok=True)
    return base_dir

def get_repo_data_dir(repo_identifier: str) -> str:
    """Get the data directory for a specific repository."""
    repo_id = normalize_repo_id(repo_identifier)
    repos_dir = _os.path.join(get_wiki_data_dir(), "repos")
    _os.makedirs(repos_dir, exist_ok=True)
    wiki_dir = _os.path.join(get_wiki_data_dir(), "wikis", repo_id)
    _os.makedirs(wiki_dir, exist_ok=True)
    return wiki_dir

def build_wiki_structure_from_requirements(repo_identifier: str, detected_types: list = None) -> list:
    """
    Deterministically generate the wiki structure based on LLM_REQUIREMENTS.md and detected repo types.
    This function is the single source of truth for the structure logic.
    """
    detected_types = detected_types or []
    sections = [
        {"id": "home-overview", "title": "Home/Overview", "description": "Purpose, Goals, Context - Synthesized from README/code analysis", "tags": ["overview", "home"], "type": "general"},
        {"id": "snapshot", "title": "Repository Snapshot / Key Statistics", "description": "File count, Languages, Comprehensive list of all AWS services used, Technologies - Analyzed from repo contents", "tags": ["snapshot", "statistics"], "type": "general"},
        {"id": "setup", "title": "Setup & Local Development", "description": "Prerequisites, Installation, Configuration, Running - Extracted from documentation, scripts, config", "tags": ["setup", "installation", "local"], "type": "general"},
        {"id": "architecture", "title": "Architecture & System Design", "description": "System structure, design choices, architecture diagram (Mermaid)", "tags": ["architecture", "design", "mermaid"], "type": "general"},
        {"id": "codebase", "title": "Codebase Structure & Key Components", "description": "Directory structure, module breakdown, entry points", "tags": ["codebase", "structure", "components"], "type": "general"},
        {"id": "build-test-deploy", "title": "Building, Testing, and Deployment", "description": "Build process, testing, deployment steps", "tags": ["build", "test", "deploy"], "type": "general"},
    ]
    if "frontend" in detected_types:
        sections.append({
            "id": "frontend", "title": "Frontend Specifics", "description": "Component overview, state management, API communication, component interaction/flow diagram (Mermaid)", "tags": ["frontend", "components", "mermaid"], "type": "frontend"
        })
    if "data-processing" in detected_types:
        sections.append({
            "id": "data-processing", "title": "Data Processing Specifics", "description": "Data sources, processing logic, outputs, data flow/sequence diagrams (Mermaid)", "tags": ["data", "processing", "mermaid"], "type": "data-processing"
        })
    if "infrastructure" in detected_types:
        sections.append({
            "id": "infrastructure", "title": "Infrastructure Specifics", "description": "Managed resources, configuration, infra diagram (Mermaid)", "tags": ["infrastructure", "iac", "mermaid"], "type": "infrastructure"
        })
    if "hybrid" in detected_types:
        sections.append({
            "id": "hybrid", "title": "Hybrid Infrastructure & Services", "description": "Covers both infrastructure and service-level design.", "tags": ["hybrid", "infrastructure", "services"], "type": "hybrid"
        })
    for t in detected_types:
        if t not in {"frontend", "data-processing", "infrastructure", "hybrid"}:
            sections.append({"id": t, "title": f"{t.title()} Specifics", "description": f"Auto-detected section for type: {t}", "tags": [t], "type": t})
    return sections

def generate_wiki_structure_with_llm(repo_identifier: str, detected_types: list = None, repo_context: str = None, use_openai: bool = False, generation_config: dict = None, repo_context_path: str = None, force_reindex: bool = False) -> list:
    """
    Generate the wiki structure using the LLM, always including the base structure but allowing the LLM to add/merge/rename up to 2 extra sections (max 8 total).
    Falls back to deterministic structure if LLM fails or output is invalid.
    
    Parameters:
        force_reindex: If True, always reindex repo files before RAG retrieval. Set to False (default) to use existing embeddings.
    """
    import subprocess
    import os
    import json as _json
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    print(f"[WIKI-STRUCTURE] Starting structure generation for {repo_identifier}")
    detected_types = detected_types or []
    base_sections = build_wiki_structure_from_requirements(repo_identifier, detected_types)
    # Gather file structure and README
    repo_dir = None
    file_list = []
    readme_content = None
    if repo_context_path and os.path.exists(repo_context_path):
        repo_dir = repo_context_path
        print(f"[WIKI-STRUCTURE] Using provided repository path: {repo_dir}")
    elif not (repo_identifier.startswith("http://") or repo_identifier.startswith("https://")):
        if os.path.exists(repo_identifier):
            repo_dir = repo_identifier
            print(f"[WIKI-STRUCTURE] Using repository identifier as path: {repo_dir}")
    # --- RAG retrieval for structure context ---
    rag_file_list = []
    if repo_dir:
        try:
            from api.langgraph.graph import run_rag_pipeline
            print(f"[WIKI-STRUCTURE] Running RAG retrieval for structure context{' with reindexing' if force_reindex else ' using existing embeddings'}...")
            rag_result = run_rag_pipeline(
                repo_identifier=repo_dir,
                query="What are the key files, modules, and documentation that define the structure and main features of this repository?",
                embedding_provider="ollama_nomic",
                generator_provider="gemini",
                top_k=20,
                skip_indexing=not force_reindex,  # Reindex if force_reindex is True
                debug=True
            )
            retrieved_docs = rag_result.get("retrieved_documents", [])
            if retrieved_docs:
                for doc in retrieved_docs:
                    file_path = doc.metadata.get("file_path")
                    if file_path and file_path not in rag_file_list:
                        rag_file_list.append(file_path)
                print(f"[WIKI-STRUCTURE] RAG selected {len(rag_file_list)} files for structure context")
        except Exception as e:
            print(f"[WIKI-STRUCTURE] RAG retrieval failed: {e}")
    # Fallback to static file list if RAG fails
    if not rag_file_list and repo_dir:
        try:
            file_list_result = subprocess.run(
                ["find", repo_dir, "-type", "f", "-not", "-path", "*/.\*", "-not", "-path", "*/node_modules/*", "-not", "-path", "*/venv/*"],
                capture_output=True,
                text=True
            )
            all_files = file_list_result.stdout.split('\n')
            file_list = [f.replace(f"{repo_dir}/", "") for f in all_files if f.strip()]
            print(f"[WIKI-STRUCTURE] Fallback: Found {len(file_list)} files in repository")
        except Exception as e:
            print(f"[WIKI-STRUCTURE] Error listing repository files: {e}")
    # Use RAG-selected files if available, else fallback
    structure_file_list = rag_file_list if rag_file_list else file_list
    readme_paths = [
        os.path.join(repo_dir, "README.md"),
        os.path.join(repo_dir, "README"),
        os.path.join(repo_dir, "README.txt"),
        os.path.join(repo_dir, "docs/README.md")
    ] if repo_dir else []
    for readme_path in readme_paths:
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                    readme_content = f.read()
                print(f"[WIKI-STRUCTURE] Found README at {readme_path}")
                break
            except Exception as e:
                print(f"[WIKI-STRUCTURE] Error reading README: {e}")
    # Compose the system prompt (do NOT append LLM_REQUIREMENTS.md)
    system_prompt = '''
You are an AI assistant specialized in generating structured outlines for technical wikis. Your task is to create a Markdown-formatted outline based on a predefined wiki structure template (which you understand) and your analysis of a provided software repository's file structure and README content.

Follow these instructions precisely:

1.  **Understand the Wiki Structure Template:** You have access to a detailed definition of the desired wiki structure template. This template defines:
    * **Standard Sections (1-6):** These sections should always be included in the wiki outline and are as follows:
        1.  Home/Overview
        2.  Repository Snapshot / Key Statistics - IMPORTANT: This section must include all AWS services used in the repository
        3.  Setup & Local Development
        4.  Architecture & System Design
        5.  Codebase Structure & Key Components
        6.  Building, Testing, and Deployment
    * **Repository Type-Specific Sections (7-9):** These sections (Frontend Specifics, Data Processing Specifics, Infrastructure Specifics) should *only* be included in the outline if your analysis of the repository context indicates they are applicable.
    * For *each* section (both standard and type-specific), the template specifies its purpose, the type of content to be derived from the repository (e.g., extracted, synthesized, inferred from code/config), and clearly indicates if a Mermaid diagram is required for that section (along with the type of diagram, e.g., Architecture, Data Flow, Component Interaction, Sequence, Infrastructure).
    * The template implies the desired order and numbering of sections in the final outline.

2.  **Analyze Repository Context:** You will receive the repository's most relevant files (selected by an AI retriever) and the content of its README file. Analyze these inputs thoroughly to determine the nature of the repository and which type-specific sections (7-9) are relevant and should be included in the outline.

3.  **Generate the Outline:** Create a Markdown outline listing the sections you have determined are applicable, following the numbering and order implied by the template.
    * Include *all* standard sections (1-6) as listed above.
    * Include only the type-specific sections (7-9) that you identified as applicable based on your analysis.
    * For each section included in the outline:
        * List the section title.
        * Briefly state its purpose (synthesized from your understanding of the template).
        * Mention the general type of content expected (e.g., "Synthesized from README/code analysis," "Extracted from documentation/scripts," "Inferred from code structure").
        * **Crucially, explicitly state if the section requires a Mermaid diagram**, mentioning the type of diagram as defined in the template (e.g., "Requires Architecture Diagram (Mermaid)", "Requires Data Flow Diagram (Mermaid)").
    * Maintain an objective, technical tone in the outline description.
4.  **Indicate Analysis Basis (Optional but Recommended):** Briefly state *why* you included certain type-specific sections (e.g., "Frontend Specifics included based on detection of [Framework/File Type] in file structure") to show the result of your analysis step.
'''
    # Insert file structure and README content
    system_prompt += """
---

**BEGIN REPOSITORY FILE STRUCTURE**
"""
    if structure_file_list:
        system_prompt += "\n".join(structure_file_list)
    else:
        system_prompt += "(No files found)"
    system_prompt += "\n**END REPOSITORY FILE STRUCTURE**\n\n---\n\n**BEGIN REPOSITORY README CONTENT**\n"
    if readme_content:
        system_prompt += readme_content
    else:
        system_prompt += "(No README found)"
    system_prompt += "\n**END REPOSITORY README CONTENT**\n\n---\n\n**TASK:** Generate the wiki outline as a JSON array of objects, each with: id, title, description, tags, and any other relevant fields. Do not include any text or Markdown outside the JSON array. Only output the JSON array. IMPORTANT: Do NOT include any numbering in the section titles. Titles should be plain, e.g., 'Home/Overview', not '1. Home/Overview'."
    print(f"[WIKI-STRUCTURE] Prompt prepared, sending to LLM...")
    if generation_config is None:
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "model_name": None
        }
    temperature = generation_config.get("temperature", 0.7)
    top_p = generation_config.get("top_p", 0.8)
    top_k = generation_config.get("top_k", 40)
    model_name = generation_config.get("model_name", None)
    try:
        if use_openai:
            from langchain_openai import ChatOpenAI
            openai_model = model_name or "gpt-4"
            print(f"[WIKI-STRUCTURE] Using OpenAI model: {openai_model}")
            llm = ChatOpenAI(
                model=openai_model,
                temperature=temperature
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            gemini_model = model_name or "gemini-2.5-flash-preview-04-17"
            print(f"[WIKI-STRUCTURE] Using Gemini model: {gemini_model}")
            llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        print(f"[WIKI-STRUCTURE] Initialized LLM, invoking chain...")
        # Use 'user' role for Gemini prompt
        print(f"[WIKI-STRUCTURE] Gemini prompt (first 500 chars):\n{system_prompt[:500]}")
        chain = ChatPromptTemplate.from_messages([("user", system_prompt)]) | llm | StrOutputParser()
        print(f"[WIKI-STRUCTURE] Sending request to model...")
        llm_response = chain.invoke({})
        print(f"[WIKI-STRUCTURE] Received response from model:\n{llm_response}")
        print(f"[WIKI-STRUCTURE] Parsing response as JSON...")
        # Remove Markdown code block markers if present
        cleaned_response = re.sub(r'^```(?:json)?\s*|```$', '', llm_response.strip(), flags=re.MULTILINE).strip()
        structure = _json.loads(cleaned_response)
        print(f"[WIKI-STRUCTURE] Successfully parsed JSON response")
        for s in structure:
            if 'type' not in s:
                s['type'] = 'general'
        print(f"[WIKI-STRUCTURE] Successfully generated structure with {len(structure)} sections")
        return structure
    except Exception as e:
        print(f"[WIKI-STRUCTURE] Error: {str(e)}. Falling back to deterministic structure.")
        return base_sections

def get_wiki_structure(repo_identifier: str, detected_types: list = None, use_llm: bool = False, repo_context: str = None, use_openai: bool = False, generation_config: dict = None, repo_context_path: str = None, force_reindex: bool = False) -> dict:
    """
    Generate the wiki structure for a repo, using a deterministic builder or LLM if use_llm=True.
    Persists the structure as /wiki-data/wikis/<repo-id>/structure.json.
    
    Parameters:
        force_reindex: If True, reindex all repo files before RAG retrieval. Set to False (default) to use existing embeddings.
    """
    detected_types = detected_types or []
    if use_llm:
        sections = generate_wiki_structure_with_llm(
            repo_identifier, 
            detected_types, 
            repo_context, 
            use_openai=use_openai,
            generation_config=generation_config,
            repo_context_path=repo_context_path,
            force_reindex=force_reindex
        )
    else:
        sections = build_wiki_structure_from_requirements(repo_identifier, detected_types)
    structure = {
        "repo": repo_identifier,
        "generated_at": _datetime.utcnow().isoformat() + "Z",
        "sections": sections
    }
    out_dir = get_repo_data_dir(repo_identifier)
    out_path = _os.path.join(out_dir, "structure.json")
    with open(out_path, "w") as f:
        _json.dump(structure, f, indent=2)
    return structure

def load_structure_from_disk(repo_identifier):
    out_dir = get_repo_data_dir(repo_identifier)
    out_path = _os.path.join(out_dir, "structure.json")
    if _os.path.exists(out_path):
        with open(out_path) as f:
            return _json.load(f)
    return None

def update_progress_file(progress_path, progress_data):
    """
    Write progress data to file with minimal lock time.
    Uses a temporary file and atomic rename to avoid file corruption.
    """
    import tempfile
    import os
    import json
    
    # Write to temporary file first
    fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='progress_tmp_', dir=os.path.dirname(progress_path))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(progress_data, f, indent=2)
        # Atomic rename to target file
        os.replace(temp_path, progress_path)
    except Exception as e:
        # Clean up temp file in case of error
        try:
            os.unlink(temp_path)
        except:
            pass
        print(f"Error updating progress file: {e}")

def generate_section_content(
    repo_identifier: str,
    detected_types: list = None,
    sections_to_generate: list = None,
    llm=None,
    force_reindex: bool = False
) -> dict:
    """
    Generate content for each wiki section using LLM. Each section is saved as a Markdown file with front matter and mermaid support.
    
    Parameters:
        force_reindex: If True, reindex repo files before RAG operations. Set to False (default) to use existing embeddings.
    """
    import os
    import yaml
    import json
    import re
    from datetime import datetime
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    print(f"[SECTION-CONTENT] Starting content generation for repository: {repo_identifier}")
    structure = load_structure_from_disk(repo_identifier)
    if not structure:
        structure = get_wiki_structure(repo_identifier, detected_types, use_llm=True, force_reindex=force_reindex)
    print(f"[SECTION-CONTENT] Got wiki structure with {len(structure['sections'])} sections")
    
    # Gather repo context (file list, README)
    repo_dir = None
    file_list = []
    readme_content = None
    if repo_identifier.startswith("http://") or repo_identifier.startswith("https://"):
        repo_dir = None  # Not supported for remote yet
    else:
        if os.path.exists(repo_identifier):
            repo_dir = repo_identifier
    if repo_dir:
        try:
            file_list = []
            for root, dirs, files in os.walk(repo_dir):
                for f in files:
                    if not f.startswith('.') and 'node_modules' not in root and 'venv' not in root:
                        rel_path = os.path.relpath(os.path.join(root, f), repo_dir)
                        file_list.append(rel_path)
        except Exception as e:
            print(f"[SECTION-CONTENT] Error listing files: {e}")
        readme_paths = [
            os.path.join(repo_dir, "README.md"),
            os.path.join(repo_dir, "README"),
            os.path.join(repo_dir, "README.txt"),
            os.path.join(repo_dir, "docs/README.md")
        ]
        for readme_path in readme_paths:
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        readme_content = f.read()
                    break
                except Exception as e:
                    print(f"[SECTION-CONTENT] Error reading README: {e}")
    
    # Output directory for pages
    out_dir = os.path.join(get_repo_data_dir(repo_identifier), "pages")
    os.makedirs(out_dir, exist_ok=True)
    
    # LLM setup (default to Gemini, can be extended)
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.7)
    
    # For progress reporting
    progress_path = os.path.join(get_repo_data_dir(repo_identifier), "progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
        except json.JSONDecodeError:
            # Fallback in case file is corrupted
            progress = {"status": "generating section content", "log": [], "sections": {}, "updated_at": datetime.utcnow().isoformat() + "Z"}
    else:
        progress = {"status": "generating section content", "log": [], "sections": {}, "updated_at": datetime.utcnow().isoformat() + "Z"}
    
    progress["status"] = "generating section content"
    progress["log"].append("Generating section content...")
    # Use atomic write instead of direct file.write
    update_progress_file(progress_path, progress)
    
    # Generate content for each section
    for i, section in enumerate(structure["sections"]):
        section_id = section["id"]
        md_path = os.path.join(out_dir, f"{section_id}.md")
        if os.path.exists(md_path):
            print(f"[SECTION-CONTENT] Skipping {section_id}, file already exists at: {md_path}")
            continue
            
        # Use specialized prompts based on section type
        base_prompt = f"""
You are an expert technical writer and code analyst. Generate a detailed Markdown page for the following wiki section, using the repository context provided.

- The page must start with YAML front matter enclosed in triple dashes (---) containing at least: title, description, tags, section_id, and generated_at.
- Example format for the YAML front matter:
  ```
  ---
  title: Section Title
  description: Brief description of the section
  tags:
    - Tag1
    - Tag2
  section_id: section-id
  generated_at: 2023-10-27T10:00:00Z
  ---
  ```
- If the section requires a diagram (see tags or description), include a mermaid code block with an appropriate diagram.
- IMPORTANT: Mermaid diagrams MUST be enclosed in proper code fences with the mermaid language identifier, like this:
  ```mermaid
  graph TD
    A["Start"] --> B["Process"]
    B --> C["End"]
  ```
- CRITICAL: Always enclose node labels in double quotes if they contain spaces or special characters:
  - DO: A["Node Label"] --> B["Another Node"]
  - DON'T: A[Node Label] --> B[Another Node]
- NEVER format mermaid diagrams like this (without code fences):
  mermaid
  graph TD
    A["Start"] --> B["Process"]
    B --> C["End"]
- CRITICAL: Do NOT include any explanatory text, comments, or descriptions inside the mermaid code block. The mermaid block should ONLY contain valid mermaid syntax. Place any explanations before or after the diagram in regular markdown text.
- INCORRECT (with text inside mermaid block):
  ```mermaid
  graph TD
    A["Start"] --> B["Process"]
    B --> C["End"]
    
    This is a simple workflow diagram showing the process flow.
  ```
- CORRECT (text outside mermaid block):
  ```mermaid
  graph TD
    A["Start"] --> B["Process"]
    B --> C["End"]
  ```
  This is a simple workflow diagram showing the process flow.
- Use information from the README and file structure as context.
- Write clear, technical, and concise documentation. Use Markdown formatting.
- Do not include any text outside the Markdown (no explanations, no code block wrappers).
- Do NOT use ```yaml or ```markdown blocks for the frontmatter - use only the triple dash format.

Section metadata:
{yaml.safe_dump(section, sort_keys=False)}

Repository file structure:
{chr(10).join(file_list[:100])}

README content:
{readme_content[:2000] if readme_content else '(No README found)'}
"""

        # Specialized prompt for snapshot section with AWS services emphasis
        if section_id == "snapshot":
            # Try to get AWS-related files using RAG
            aws_files = []
            try:
                from api.langgraph.graph import run_rag_pipeline
                aws_query = "Find all files related to AWS services, including CloudFormation templates, Terraform files, AWS SDK usage, IAM policies, and configuration files for AWS services"
                print(f"[SECTION-CONTENT] Running AWS service discovery{' with reindexing' if force_reindex else ' using existing embeddings'}...")
                rag_result = run_rag_pipeline(
                    repo_identifier=repo_dir if repo_dir else repo_identifier,
                    query=aws_query,
                    embedding_provider="ollama_nomic",
                    skip_indexing=not force_reindex,  # Reindex if force_reindex is True
                    top_k=15
                )
                if rag_result and "retrieved_documents" in rag_result:
                    aws_files = [doc.metadata.get("file_path") for doc in rag_result["retrieved_documents"] if doc.metadata.get("file_path")]
                    print(f"[SECTION-CONTENT] Found {len(aws_files)} AWS-related files through RAG")
            except Exception as e:
                print(f"[SECTION-CONTENT] Error retrieving AWS files: {e}")

            prompt = f"""
You are an expert technical writer and AWS cloud architect. Generate a detailed Markdown page for the "Repository Snapshot / Key Statistics" wiki section, with a strong focus on AWS services used.

- The page must start with YAML front matter enclosed in triple dashes (---) containing at least: title, description, tags, section_id, and generated_at.
- Example format for the YAML front matter:
  ```
  ---
  title: Repository Snapshot / Key Statistics
  description: Brief description of the section
  tags:
    - Tag1
    - Tag2
  section_id: section-id
  generated_at: 2023-10-27T10:00:00Z
  ---
  ```
- Structure your response with clear headers and bullet points.
- CRITICAL: Identify and list ALL AWS services used in this repository. For each AWS service:
  * Include the service name
  * Describe how it's used in this project
  * Note any specific configurations or implementations
  * Group related services together when appropriate
- Also include general repository statistics like:
  * File count by type/language
  * Key technologies and frameworks used
  * Main programming languages
  * Repository size and structure summary
- Use a table format where appropriate.
- If possible, include a Mermaid diagram showing how AWS services interact.
- IMPORTANT: Mermaid diagrams MUST be enclosed in proper code fences with the mermaid language identifier, like this:
  ```mermaid
  graph TD
    A["Lambda"] --> B["DynamoDB"]
    C["API Gateway"] --> A
  ```
- CRITICAL: Always enclose node labels in double quotes if they contain spaces or special characters:
  - DO: A["Lambda Function"] --> B["DynamoDB Table"]
  - DON'T: A[Lambda Function] --> B[DynamoDB Table]
- CRITICAL: Do NOT include any explanatory text, comments, or descriptions inside the mermaid code block. The mermaid block should ONLY contain valid mermaid syntax. Place any explanations before or after the diagram in regular markdown text.
- INCORRECT (with text inside mermaid block):
  ```mermaid
  graph TD
    A["Lambda"] --> B["DynamoDB"]
    
    This diagram shows how Lambda connects to DynamoDB.
  ```
- CORRECT (text outside mermaid block):
  ```mermaid
  graph TD
    A["Lambda"] --> B["DynamoDB"]
    C["API Gateway"] --> A
  ```
  This diagram shows how Lambda connects to DynamoDB and receives requests from API Gateway.
- NEVER format mermaid diagrams like this (without code fences):
  mermaid
  graph TD
    A["Lambda"] --> B["DynamoDB"]
    C["API Gateway"] --> A
- Use information from the README and file structure as context.
- Write clear, technical, and concise documentation using proper Markdown formatting.
- Do not include any text outside the Markdown (no explanations, no code block wrappers).
- Do NOT use ```yaml or ```markdown blocks for the frontmatter - use only the triple dash format.

Section metadata:
{yaml.safe_dump(section, sort_keys=False)}

Repository file structure:
{chr(10).join(file_list[:100])}

README content:
{readme_content[:2000] if readme_content else '(No README found)'}

AWS-Related Files:
{chr(10).join(aws_files) if aws_files else '(No AWS-specific files identified)'}
"""
        else:
            prompt = base_prompt
        
        chain = ChatPromptTemplate.from_messages([("user", prompt)]) | llm | StrOutputParser()
        try:
            print(f"[SECTION-CONTENT] Generating content for section: {section_id} ({i+1}/{len(structure['sections'])})")
            md_content = chain.invoke({})
            
            # Post-process the content to ensure consistent frontmatter format
            # 1. First remove any code block wrappers if present
            md_content = re.sub(r'^```(?:markdown|yaml)?(?:\s*\n)?|```$', '', md_content.strip(), flags=re.MULTILINE)
            
            # 2. Convert any ```yaml frontmatter to --- format
            yaml_pattern = re.compile(r'^```yaml\s*\n([\s\S]*?)(?:\n```|$)', re.MULTILINE)
            yaml_match = yaml_pattern.search(md_content)
            if yaml_match:
                yaml_content = yaml_match.group(1)
                md_content = yaml_pattern.sub(f'---\n{yaml_content}\n---\n', md_content)
                print(f"[SECTION-CONTENT] Converted ```yaml frontmatter to --- format for {section_id}")
            
            # 3. Ensure there's at least one empty line after frontmatter
            md_content = re.sub(r'---\s*\n', '---\n\n', md_content)

            # 4. Fix incorrectly formatted mermaid diagrams (without code fence)
            # Look for "mermaid\ngraph" pattern which indicates incorrectly formatted diagrams
            mermaid_pattern = re.compile(r'\n(mermaid)\s*\n(graph\s+[A-Z][A-Z][\s\S]*?)(?=\n##|\n#|\Z)', re.MULTILINE)
            if mermaid_pattern.search(md_content):
                md_content = mermaid_pattern.sub(r'\n```mermaid\n\2\n```\n', md_content)
                print(f"[SECTION-CONTENT] Fixed mermaid diagram format for {section_id}")

            # Handle other common mermaid diagram types
            sequence_pattern = re.compile(r'\n(mermaid)\s*\n(sequenceDiagram[\s\S]*?)(?=\n##|\n#|\Z)', re.MULTILINE)
            if sequence_pattern.search(md_content):
                md_content = sequence_pattern.sub(r'\n```mermaid\n\2\n```\n', md_content)
                print(f"[SECTION-CONTENT] Fixed sequence diagram format for {section_id}")

            flowchart_pattern = re.compile(r'\n(mermaid)\s*\n(flowchart\s+[A-Z][A-Z][\s\S]*?)(?=\n##|\n#|\Z)', re.MULTILINE)
            if flowchart_pattern.search(md_content):
                md_content = flowchart_pattern.sub(r'\n```mermaid\n\2\n```\n', md_content)
                print(f"[SECTION-CONTENT] Fixed flowchart diagram format for {section_id}")

            # Check for incomplete mermaid diagrams (no ending ```)
            mermaid_blocks = re.findall(r'```mermaid\n[\s\S]*?(?=```|\n##|\n#|\Z)', md_content)
            for block in mermaid_blocks:
                if not block.endswith('```'):
                    fixed_block = block + '\n```'
                    md_content = md_content.replace(block, fixed_block)
                    print(f"[SECTION-CONTENT] Added missing closing fence to mermaid diagram in {section_id}")
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            if "sections" not in progress:
                progress["sections"] = {}
            
            progress["sections"][section_id] = "done"
            progress["log"].append(f"Section {section_id} generated.")
            # Use atomic write
            update_progress_file(progress_path, progress)
        except Exception as e:
            error_message = str(e)
            print(f"[SECTION-CONTENT] Error generating section {section_id}: {error_message}")
            
            if "sections" not in progress:
                progress["sections"] = {}
                
            progress["sections"][section_id] = f"error: {error_message}"
            progress["log"].append(f"Error generating section {section_id}: {error_message}")
            # Use atomic write
            update_progress_file(progress_path, progress)
    
    progress["status"] = "done"
    progress["log"].append("Wiki page generation complete.")
    progress["finished_at"] = datetime.utcnow().isoformat() + "Z"
    # Use atomic write
    update_progress_file(progress_path, progress)
    
    return {"status": "done", "sections": structure["sections"]}

# --- Usage Example ---
if __name__ == "__main__":
    repo = "https://github.com/example/repo"
    print("Wiki structure:")
    print(get_wiki_structure(repo)) 