import os
import logging
from fastapi import FastAPI, HTTPException, Request, Body, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, PlainTextResponse
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
import google.generativeai as genai
from api.langgraph.graph import run_rag_pipeline
from api.langgraph.wiki_structure import get_wiki_structure, generate_section_content, get_repo_data_dir, update_progress_file
from starlette.status import HTTP_404_NOT_FOUND
import shutil
import subprocess
from api.langgraph.chroma_utils import generate_collection_name, get_chroma_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API keys from environment variables
google_api_key = os.environ.get('GOOGLE_API_KEY')

# Configure Google Generative AI
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

# Initialize FastAPI app
app = FastAPI(
    title="Streaming API",
    description="API for streaming chat completions"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str
    relatedPages: List[str]

class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    Export wiki content as Markdown or JSON.

    Args:
        request: The export request containing wiki pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # Generate Markdown content
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            # Generate JSON content
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # Create response with appropriate headers for file download
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate Markdown export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"

        # Add related files
        if page.filePaths and len(page.filePaths) > 0:
            markdown += "### Related Files\n\n"
            for file_path in page.filePaths:
                markdown += f"- `{file_path}`\n"
            markdown += "\n"

        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate JSON export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        JSON content as string
    """
    # Create a dictionary with metadata and pages
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # Convert to JSON string with pretty formatting
    return json.dumps(export_data, indent=2)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": {
            "Chat": [
                "POST /chat/completions/stream - Streaming chat completion",
            ],
            "Wiki": [
                "POST /export/wiki - Export wiki content as Markdown or JSON",
            ]
        }
    }

class AnalyseRepositoryRequest(BaseModel):
    repo_url: str
    messages: List[Dict[str, str]]
    filePath: Optional[str] = None
    use_ollama: Optional[bool] = False
    top_k: Optional[int] = None
    skip_indexing: Optional[bool] = True

class AnalyseRepositoryResponse(BaseModel):
    answer: str
    metadata: Dict[str, Any]
    retrieved_documents: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

@app.post("/analyse-repository", response_model=AnalyseRepositoryResponse)
async def analyse_repository(request: AnalyseRepositoryRequest = Body(...)):
    """
    Analyse a repository and answer a question using the LangGraph RAG pipeline.
    """
    try:
        # Use the last user message as the query
        user_messages = [m for m in request.messages if m.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found in 'messages'.")
        query = user_messages[-1]["content"]

        # Call the RAG pipeline
        result = run_rag_pipeline(
            repo_identifier=request.repo_url,
            query=query,
            use_ollama=request.use_ollama or False,
            top_k=request.top_k,
            debug=True,  # For now, use debug pipeline
            skip_indexing=request.skip_indexing if request.skip_indexing is not None else True
        )
        return AnalyseRepositoryResponse(
            answer=result.get("answer", ""),
            metadata=result.get("metadata", {}),
            retrieved_documents=[doc.metadata for doc in result.get("retrieved_documents", [])] if result.get("retrieved_documents") else None,
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Error in /analyse-repository: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wiki-structure")
async def wiki_structure(
    repo_url: str = Query(..., description="Repository URL or path"),
    detected_types: Optional[str] = Query(None, description="Comma-separated list of detected types"),
    use_llm: bool = Query(True, description="Whether to use LLM-powered structure generation (default: True)"),
    force_reindex: bool = Query(False, description="Whether to force reindexing all repo files (default: False)")
):
    """
    Return the generated wiki structure for a given repo. If not present, generates it.
    detected_types: optional, comma-separated (e.g. 'frontend,infrastructure')
    use_llm: optional, bool (default: True) - whether to use LLM-powered structure generation
    force_reindex: optional, bool (default: False) - whether to force reindexing all repo files
    """
    types = [t.strip() for t in detected_types.split(",")] if detected_types else []
    from api.langgraph.wiki_structure import load_structure_from_disk, get_wiki_structure, get_wiki_data_dir
    
    # Check if repo_url is actually a normalized repo ID (directory name)
    import os
    wikis_dir = os.path.join(get_wiki_data_dir(), "wikis")
    direct_wiki_path = os.path.join(wikis_dir, repo_url)
    
    if os.path.isdir(direct_wiki_path):
        # If it's a wiki folder name, load directly from that
        structure_path = os.path.join(direct_wiki_path, "structure.json")
        if os.path.exists(structure_path):
            with open(structure_path, 'r') as f:
                import json
                return json.load(f)
    
    # If not a direct wiki folder or structure doesn't exist, proceed normally
    structure = load_structure_from_disk(repo_url)
    if not structure:
        structure = get_wiki_structure(
            repo_url, 
            detected_types=types, 
            use_llm=True,
            use_openai=False,  # Always use Gemini
            generation_config=None,
            force_reindex=force_reindex
        )
    return structure

def safe_load_progress(progress_path):
    try:
        with open(progress_path) as f:
            return json.load(f)
    except Exception as e:
        return {"status": "error", "log": [f"Error reading progress file: {e}"], "error": str(e)}

@app.get("/wiki-progress")
async def wiki_progress(repo_url: str = Query(..., description="Repository URL or path"), detected_types: Optional[str] = Query(None, description="Comma-separated list of detected types")):
    """
    Return the progress.json for a given repo. Non-blocking implementation that returns the latest progress data.
    """
    types = [t.strip() for t in detected_types.split(",")] if detected_types else []
    import os
    from api.langgraph.wiki_structure import get_repo_data_dir, get_wiki_data_dir
    
    # Check if repo_url is actually a normalized repo ID (directory name)
    wikis_dir = os.path.join(get_wiki_data_dir(), "wikis")
    direct_wiki_path = os.path.join(wikis_dir, repo_url)
    
    if os.path.isdir(direct_wiki_path):
        # If the repo_url is actually a directory name in wikis/, use it directly
        repo_dir = direct_wiki_path
    else:
        # Otherwise normalize it to get the repo directory
        repo_dir = get_repo_data_dir(repo_url)
        
    progress_path = os.path.join(repo_dir, "progress.json")
    
    # Just return current progress file if it exists (non-blocking)
    if os.path.exists(progress_path):
        try:
            import fcntl
            import time
            
            def read_file_nonblocking(file_path, max_retries=3, retry_delay=0.05):
                """Attempt to read a file without blocking, using non-blocking file locking."""
                for attempt in range(max_retries):
                    try:
                        with open(file_path, "r") as f:
                            # Try to get a shared (read) lock, but don't block
                            try:
                                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                                data = f.read()
                                fcntl.flock(f, fcntl.LOCK_UN)
                                return data
                            except IOError:
                                # File is locked, return None
                                if attempt < max_retries - 1:
                                    time.sleep(retry_delay)
                                    continue
                                return None
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            raise e
                return None
            
            # Try to read non-blocking first
            file_content = read_file_nonblocking(progress_path)
            if file_content is not None:
                try:
                    return json.loads(file_content)
                except json.JSONDecodeError:
                    # If we got partial JSON, return a simple status
                    return {"status": "processing", "log": ["Processing repository..."], "error": "Progress file is being updated"}
            
            # If non-blocking read failed (file is locked), return a simple status
            return {"status": "processing", "log": ["Processing repository..."], "error": "Progress file is currently locked"}
            
        except Exception as e:
            # Return simple progress if file is being written to or corrupted
            return {"status": "processing", "log": ["Processing repository..."], "error": f"Could not read progress file: {str(e)}"}
    
    # If no progress file exists yet
    return {"status": "not_started", "log": ["Wiki generation not started yet."]}

@app.get("/wiki-section")
async def wiki_section(repo_url: str = Query(..., description="Repository URL or path"), section_id: str = Query(..., description="Section ID"), detected_types: Optional[str] = Query(None, description="Comma-separated list of detected types")):
    """
    Return the Markdown content for a given section. 404 if not found.
    """
    import os
    from api.langgraph.wiki_structure import get_repo_data_dir, get_wiki_data_dir, normalize_repo_id
    
    # Check if repo_url is actually a normalized repo ID (directory name)
    wikis_dir = os.path.join(get_wiki_data_dir(), "wikis")
    direct_wiki_path = os.path.join(wikis_dir, repo_url)
    
    if os.path.isdir(direct_wiki_path):
        # If the repo_url is actually a directory name in wikis/, use it directly
        repo_dir = direct_wiki_path
    else:
        # Otherwise normalize it to get the repo directory
        repo_dir = get_repo_data_dir(repo_url)
    
    md_path = os.path.join(repo_dir, "pages", f"{section_id}.md")
    
    if not os.path.exists(md_path):
        return PlainTextResponse("Section not found", status_code=HTTP_404_NOT_FOUND)
    with open(md_path) as f:
        return PlainTextResponse(f.read(), media_type="text/markdown")

@app.post("/start-wiki-generation")
async def start_wiki_generation(
    repo_url: str = Body(..., embed=True, description="Repository URL or local path"),
    detected_types: Optional[str] = Body(None, embed=True, description="Comma-separated list of detected types"),
    model: str = Body("gemini", embed=True, description="Model provider to use: 'gemini', 'openai', or 'deterministic'"),
    model_name: Optional[str] = Body(None, embed=True, description="Specific model name (e.g., 'gemini-1.5-flash', 'gpt-4')"),
    temperature: Optional[float] = Body(0.7, embed=True, description="Temperature for generation (0.0-1.0)"),
    top_p: Optional[float] = Body(0.8, embed=True, description="Top-p sampling parameter (0.0-1.0)"),
    top_k: Optional[int] = Body(40, embed=True, description="Top-k sampling parameter"),
    force_reindex: Optional[bool] = Body(False, embed=True, description="Whether to force reindexing all repo files (default: False)"),
    background_tasks: BackgroundTasks = None
):
    """
    Orchestrate the full wiki generation pipeline for a repository:
    1. Scan/clone/copy repo
    2. Generate wiki structure (LLM or deterministic)
    3. Generate section content/pages
    4. Write progress at each step
    
    Parameters:
    - repo_url: Repository URL or local path
    - detected_types: Optional comma-separated list of repo types
    - model: Model provider to use ('gemini', 'openai', or 'deterministic')
    - model_name: Specific model name (e.g., 'gemini-1.5-flash', 'gpt-4')
    - temperature: Temperature for generation (0.0-1.0)
    - top_p: Top-p sampling parameter (0.0-1.0)
    - top_k: Top-k sampling parameter
    - force_reindex: Whether to force reindexing all repo files (default: False)
    """
    import os
    import json
    import shutil
    import subprocess
    from api.langgraph.wiki_structure import normalize_repo_id, get_wiki_structure, generate_section_content
    from api.langgraph.wiki_structure import get_repo_data_dir, get_wiki_data_dir
    from datetime import datetime
    
    # Validate model parameter
    valid_models = ["gemini", "openai", "deterministic"]
    if model not in valid_models:
        return {"error": f"Invalid model. Must be one of: {', '.join(valid_models)}"}, 400
    
    # Set default model names if not specified
    if model_name is None:
        if model == "gemini":
            model_name = "gemini-2.5-flash-preview-04-17"
        elif model == "openai":
            model_name = "gpt-4"
    
    # Validate temperature
    if temperature is not None and (temperature < 0.0 or temperature > 1.0):
        return {"error": "Temperature must be between 0.0 and 1.0"}, 400
    
    # Validate top_p
    if top_p is not None and (top_p < 0.0 or top_p > 1.0):
        return {"error": "Top-p must be between 0.0 and 1.0"}, 400
    
    types = [t.strip() for t in detected_types.split(",")] if detected_types else []
    repo_id = normalize_repo_id(repo_url)
    
    # Use new directory structure
    repos_dir = os.path.join(get_wiki_data_dir(), "repos")
    os.makedirs(repos_dir, exist_ok=True)
    
    wiki_dir = get_repo_data_dir(repo_url)
    progress_path = os.path.join(wiki_dir, "progress.json")
    
    # Create model_config for logging and passing to generation functions
    model_config = {
        "provider": model,
        "name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }
    
    # Start progress
    progress = {
        "status": "starting", 
        "log": ["Starting wiki generation..."], 
        "started_at": datetime.utcnow().isoformat() + "Z", 
        "model": model,
        "model_config": model_config
    }
    
    update_progress_file(progress_path, progress)
        
    def pipeline():
        try:
            # --- Clean up wiki data before reindexing ---
            wiki_data_dir = os.path.join("wiki-data", "wikis", repo_id)
            if os.path.exists(wiki_data_dir) and force_reindex:
                try:
                    shutil.rmtree(wiki_data_dir)
                    print(f"Deleted existing wiki data for {repo_id} before reindexing.")
                except Exception as e:
                    print(f"Error deleting wiki data for {repo_id}: {e}")
            
            # Make sure wiki directories exist
            os.makedirs(wiki_data_dir, exist_ok=True)
            pages_dir = os.path.join(wiki_data_dir, "pages")
            os.makedirs(pages_dir, exist_ok=True)
            
            # Check if content already exists and we're not forcing reindex
            if not force_reindex and os.path.exists(os.path.join(wiki_data_dir, "structure.json")):
                # Check if pages directory has content
                if os.listdir(pages_dir):
                    progress["status"] = "done"
                    progress["log"].append("Wiki content already exists. Skipping regeneration.")
                    progress["finished_at"] = datetime.utcnow().isoformat() + "Z"
                    update_progress_file(progress_path, progress)
                    return progress
            
            # Step 1: Clone/copy repo if needed
            progress["status"] = "cloning/copying repository"
            progress["log"].append(f"Starting clone/copy step for repo: {repo_url}")
            update_progress_file(progress_path, progress)
            repo_is_git = repo_url.startswith("http://") or repo_url.startswith("https://") or repo_url.startswith("git@")
            repo_dest = os.path.join(repos_dir, repo_id)
            wiki_data_dir = os.path.join("wiki-data", "wikis", repo_id)
            if not os.path.exists(repo_dest):
                progress["log"].append(f"{'Cloning' if repo_is_git else 'Copying'} repository to {repo_dest}...")
                update_progress_file(progress_path, progress)
                if repo_is_git:
                    try:
                        subprocess.run(["git", "clone", repo_url, repo_dest], check=True)
                    except Exception as e:
                        progress["status"] = "error"
                        progress["log"].append(f"Error cloning repo: {e}")
                        update_progress_file(progress_path, progress)
                        return progress
                else:
                    shutil.copytree(repo_url, repo_dest)
            # --- Clean up ChromaDB collection before reindexing ---
            collection_name = generate_collection_name(repo_dest)
            client = get_chroma_client()
            
            # Check if collection exists first
            collection_exists = False
            try:
                collections = client.list_collections()
                collection_exists = any(c.name == collection_name for c in collections)
                print(f"ChromaDB collection check: '{collection_name}' {'exists' if collection_exists else 'does not exist'}")
            except Exception as e:
                print(f"Error checking ChromaDB collections: {e}")
            
            # Only delete the collection if force_reindex is True AND collection exists
            if force_reindex and collection_exists:
                try:
                    print(f"Deleting existing ChromaDB collection '{collection_name}' because force_reindex=True.")
                    client.delete_collection(collection_name)
                    # Wait/retry to ensure deletion
                    import time
                    for _ in range(10):
                        collections = client.list_collections()
                        if not any(c.name == collection_name for c in collections):
                            break
                        time.sleep(0.2)
                    else:
                        print(f"Warning: Collection '{collection_name}' still exists after repeated deletion attempts.")
                except Exception as e:
                    print(f"Warning: Failed to delete ChromaDB collection '{collection_name}': {e}")
            elif not collection_exists:
                print(f"Note: No existing collection found. Will need to create embeddings even though force_reindex=False.")
            elif not force_reindex:
                print(f"Note: Using existing collection '{collection_name}' (force_reindex=False).")
                    
            # --- RAG pipeline trigger follows ---
            # The first run will always create embeddings regardless of force_reindex setting
            progress["status"] = "indexing (RAG pipeline)"
            if not collection_exists:
                progress["log"].append("Starting RAG pipeline with indexing (first-time indexing required)...")
            else:
                progress["log"].append(f"{'Starting' if force_reindex else 'Running'} RAG pipeline{' (with reindexing)' if force_reindex else ' (using existing embeddings)'}...")
            update_progress_file(progress_path, progress)
            try:
                # Add a guard to ensure store_vectors_node is only called once per pipeline run
                from api.langgraph.graph import run_rag_pipeline
                pipeline_state = {"store_vectors_called": False}
                def guarded_store_vectors_node(state):
                    if pipeline_state["store_vectors_called"]:
                        print("store_vectors_node already called for this pipeline run, skipping.")
                        return state
                    pipeline_state["store_vectors_called"] = True
                    from api.langgraph.nodes.store_vectors import store_vectors_node
                    return store_vectors_node(state)
                # Patch the graph to use the guarded node
                import api.langgraph.graph as graph_mod
                graph_mod.store_vectors_node = guarded_store_vectors_node
                
                print(f"Calling RAG pipeline with skip_indexing={force_reindex is False}")
                run_rag_pipeline(
                    repo_dest,
                    "What are the key files, modules, and documentation that define the structure, architecture, and main features of this repository? Include files that are essential for understanding how the project is organized and how its main components interact.",
                    embedding_provider="ollama_nomic",
                    skip_indexing=not force_reindex  # Note: If collection doesn't exist, run_rag_pipeline will index anyway
                )
            except Exception as e:
                progress["status"] = "error"
                progress["log"].append(f"Error running RAG pipeline: {e}")
                update_progress_file(progress_path, progress)
                return progress

            progress["status"] = "scanning repository"
            progress["log"].append("Scanning repository...")
            update_progress_file(progress_path, progress)
            # Step 2: Generate wiki structure (using specified model)
            model_desc = f"{model_name} (temp={temperature}, top_p={top_p})" if model != "deterministic" else "deterministic"
            progress["status"] = f"generating wiki structure ({model_desc})"
            progress["log"].append(f"Starting wiki structure generation using {model_desc}...")
            update_progress_file(progress_path, progress)
            if not os.path.exists(repo_dest):
                progress["status"] = "error"
                progress["log"].append(f"Error: Repository path not found at {repo_dest}")
                update_progress_file(progress_path, progress)
                return
            use_llm = model != "deterministic"
            use_openai = model == "openai"
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "model_name": model_name
            }
            try:
                structure = get_wiki_structure(
                    repo_url,
                    detected_types=types,
                    use_llm=use_llm,
                    use_openai=use_openai if use_llm else False,
                    generation_config=generation_config,
                    repo_context_path=repo_dest,
                    force_reindex=force_reindex
                )
                progress["status"] = "wiki structure generated"
                progress["log"].append(f"Wiki structure generated using {model_desc}.")
                update_progress_file(progress_path, progress)
            except Exception as e:
                progress["status"] = "error"
                progress["log"].append(f"Error generating wiki structure: {str(e)}")
                update_progress_file(progress_path, progress)
                return
            # Step 3: Generate section content
            progress["status"] = "generating section content"
            progress["log"].append("Starting section content generation...")
            update_progress_file(progress_path, progress)
            try:
                generate_section_content(repo_url, types, force_reindex=force_reindex)
                progress["status"] = "done"
                progress["log"].append("Wiki generation complete.")
                progress["finished_at"] = datetime.utcnow().isoformat() + "Z"
                update_progress_file(progress_path, progress)
            except Exception as e:
                progress["status"] = "error"
                progress["log"].append(f"Error generating section content: {str(e)}")
                update_progress_file(progress_path, progress)
                return progress
        except Exception as e:
            progress["status"] = "error"
            progress["log"].append(f"Error: {str(e)}")
            update_progress_file(progress_path, progress)
                
    if background_tasks is not None:
        background_tasks.add_task(pipeline)
    else:
        pipeline()
        
    return {
        "status": "started", 
        "repo_id": repo_id, 
        "progress_path": progress_path, 
        "model": model,
        "model_config": model_config
    }

@app.get("/list-wikis")
async def list_wikis():
    """
    Return a list of all wikis that have been generated.
    """
    import os
    from api.langgraph.wiki_structure import get_wiki_data_dir
    
    wikis_dir = os.path.join(get_wiki_data_dir(), "wikis")
    os.makedirs(wikis_dir, exist_ok=True)
    
    wikis = []
    try:
        for wiki_id in os.listdir(wikis_dir):
            wiki_path = os.path.join(wikis_dir, wiki_id)
            if os.path.isdir(wiki_path):
                # Check if this wiki has a structure.json file (indicator of a complete wiki)
                structure_path = os.path.join(wiki_path, "structure.json")
                has_structure = os.path.exists(structure_path)
                
                # Check if pages directory exists and has content
                pages_dir = os.path.join(wiki_path, "pages")
                has_pages = os.path.exists(pages_dir) and os.listdir(pages_dir)
                
                # Get progress info to check status
                progress_path = os.path.join(wiki_path, "progress.json")
                status = "unknown"
                if os.path.exists(progress_path):
                    try:
                        with open(progress_path, "r") as f:
                            progress = json.load(f)
                            status = progress.get("status", "unknown")
                    except:
                        pass
                
                # Only include wikis that are either complete or in progress
                if has_structure or status != "unknown":
                    # Get repo path if available
                    repo_path = ""
                    repos_dir = os.path.join(get_wiki_data_dir(), "repos")
                    if os.path.exists(os.path.join(repos_dir, wiki_id)):
                        repo_path = os.path.join(repos_dir, wiki_id)
                    
                    wikis.append({
                        "id": wiki_id,
                        "name": wiki_id,
                        "path": repo_path,
                        "status": status,
                        "has_structure": has_structure,
                        "has_pages": has_pages,
                        "wiki_path": wiki_path,
                    })
    except Exception as e:
        print(f"Error listing wikis: {e}")
    
    return {"wikis": wikis}
