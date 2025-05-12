import os as _os
import json as _json
from datetime import datetime as _datetime
import re

def normalize_repo_id(repo_identifier: str) -> str:
    """Create a safe directory name for the repo."""
    if repo_identifier.startswith("http://") or repo_identifier.startswith("https://"):
        slug = repo_identifier.rstrip("/").split("/")[-1].replace(".git", "")
        owner = repo_identifier.rstrip("/").split("/")[-2]
        base = f"{owner}_{slug}"
    else:
        base = _os.path.basename(_os.path.abspath(repo_identifier))
    
    # Replace ALL non-alphanumeric characters with underscores
    base = re.sub(r'[^a-zA-Z0-9]', '_', base)
    
    return base

def build_wiki_structure_from_requirements(repo_identifier: str, detected_types: list = None) -> list:
    """
    Deterministically generate the wiki structure based on LLM_REQUIREMENTS.md and detected repo types.
    This function is the single source of truth for the structure logic.
    """
    detected_types = detected_types or []
    sections = [
        {"id": "overview", "title": "Home/Overview", "description": "Purpose, Goals, Context - Synthesized from README/code analysis", "tags": ["overview", "home"], "type": "general"},
        {"id": "snapshot", "title": "Repository Snapshot / Key Statistics", "description": "File count, Languages, AWS services, Technologies - Analyzed from repo contents", "tags": ["snapshot", "statistics"], "type": "general"},
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

def generate_wiki_structure_with_llm(repo_identifier: str, detected_types: list = None, repo_context: str = None, use_openai: bool = False, generation_config: dict = None) -> list:
    """
    Generate the wiki structure using the LLM, always including the base structure but allowing the LLM to add/merge/rename up to 2 extra sections (max 8 total).
    Falls back to deterministic structure if LLM fails or output is invalid.
    
    Parameters:
    - repo_identifier: Repository URL or path
    - detected_types: Optional list of detected repository types
    - repo_context: Optional context about the repository
    - use_openai: Whether to use OpenAI (True) or Google Gemini (False)
    - generation_config: Dict with configuration parameters:
        - temperature: Temperature for generation (0.0-1.0) 
        - top_p: Top-p sampling parameter (0.0-1.0)
        - top_k: Top-k sampling parameter
        - model_name: Specific model name to use
    """
    detected_types = detected_types or []
    base_sections = build_wiki_structure_from_requirements(repo_identifier, detected_types)
    # Compose the prompt
    base_titles = [s['title'] for s in base_sections]
    prompt = f"""
You are an expert technical writer and code analyst.
Given the following repository and its base wiki structure, suggest a final wiki structure (max 8 sections).
- Always include the base sections below.
- You may add, merge, or rename sections if it improves clarity or coverage for this repo.
- Suggest at most 2 additional sections if they are highly relevant.
- Output the final structure as a JSON array of objects, each with: id, title, description, and tags.

Repository: {repo_identifier}
Base structure:
"""
    for i, s in enumerate(base_sections, 1):
        prompt += f"{i}. {s['title']}\n"
    prompt += "\n(You may add e.g. 'Frontend Specifics', 'Data Processing', 'Infrastructure', or other relevant sections.)\n"
    if repo_context:
        prompt += f"\nRepository context (README, file list, etc.):\n{repo_context}\n"
    prompt += "\nRespond with only the JSON array."

    # Set default generation config if not provided
    if generation_config is None:
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "model_name": None
        }
    
    # Extract config values or use defaults
    temperature = generation_config.get("temperature", 0.7)
    top_p = generation_config.get("top_p", 0.8)
    top_k = generation_config.get("top_k", 40)
    model_name = generation_config.get("model_name", None)
    
    # Call the LLM (Gemini or OpenAI)
    try:
        if use_openai:
            from langchain_openai import ChatOpenAI
            # Use specified model name or default to gpt-4
            openai_model = model_name or "gpt-4"
            llm = ChatOpenAI(
                model=openai_model, 
                temperature=temperature
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Use specified model name or default to gemini-2.5-flash-preview-04-17
            gemini_model = model_name or "gemini-2.5-flash-preview-04-17"
            llm = ChatGoogleGenerativeAI(
                model=gemini_model, 
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        chain = ChatPromptTemplate.from_messages([("system", prompt)]) | llm | StrOutputParser()
        llm_response = chain.invoke({})
        # Try to parse the JSON array
        structure = _json.loads(llm_response)
        # Validate: must include all base section titles, max 8 total
        titles = {s['title'].lower() for s in structure}
        missing = [t for t in base_titles if t.lower() not in titles]
        if missing or not (1 <= len(structure) <= 8):
            raise ValueError(f"LLM structure missing required sections or too many sections: missing={missing}, count={len(structure)}")
        # Add 'type' field if missing
        for s in structure:
            if 'type' not in s:
                s['type'] = 'general'
        return structure
    except Exception as e:
        print(f"[LLM Structure Generation] Error or invalid output: {e}. Falling back to deterministic structure.")
        return base_sections

def get_wiki_structure(repo_identifier: str, detected_types: list = None, use_llm: bool = False, repo_context: str = None, use_openai: bool = False, generation_config: dict = None) -> dict:
    """
    Generate the wiki structure for a repo, using a deterministic builder or LLM if use_llm=True.
    Persists the structure as /wiki-data/<repo-id>/structure.json.
    
    Parameters:
    - repo_identifier: Repository URL or path
    - detected_types: Optional list of detected repository types
    - use_llm: Whether to use LLM for structure generation
    - repo_context: Optional context about the repository
    - use_openai: Whether to use OpenAI (True) or Google Gemini (False)
    - generation_config: Dict with model configuration parameters (temperature, top_p, top_k, model_name)
    """
    detected_types = detected_types or []
    if use_llm:
        sections = generate_wiki_structure_with_llm(
            repo_identifier, 
            detected_types, 
            repo_context, 
            use_openai=use_openai,
            generation_config=generation_config
        )
    else:
        sections = build_wiki_structure_from_requirements(repo_identifier, detected_types)
    structure = {
        "repo": repo_identifier,
        "generated_at": _datetime.utcnow().isoformat() + "Z",
        "sections": sections
    }
    repo_id = normalize_repo_id(repo_identifier)
    out_dir = _os.path.join(_os.path.dirname(__file__), "..", "wiki-data", repo_id)
    _os.makedirs(out_dir, exist_ok=True)
    out_path = _os.path.join(out_dir, "structure.json")
    with open(out_path, "w") as f:
        _json.dump(structure, f, indent=2)
    return structure 