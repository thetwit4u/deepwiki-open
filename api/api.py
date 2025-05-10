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
from api.langgraph.wiki_structure import get_wiki_structure, generate_section_content, get_repo_data_dir
from starlette.status import HTTP_404_NOT_FOUND
import shutil
import subprocess

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
    use_llm: bool = Query(True, description="Whether to use LLM-powered structure generation (default: True)")
):
    """
    Return the generated wiki structure for a given repo. If not present, generates it.
    detected_types: optional, comma-separated (e.g. 'frontend,infrastructure')
    use_llm: optional, bool (default: True) - whether to use LLM-powered structure generation
    """
    types = [t.strip() for t in detected_types.split(",")] if detected_types else []
    from api.langgraph.wiki_structure import load_structure_from_disk, get_wiki_structure
    structure = load_structure_from_disk(repo_url)
    if not structure:
        structure = get_wiki_structure(
            repo_url, 
            detected_types=types, 
            use_llm=True,
            use_openai=False,  # Always use Gemini
            generation_config=None
        )
    return structure

@app.get("/wiki-progress")
async def wiki_progress(repo_url: str = Query(..., description="Repository URL or path"), detected_types: Optional[str] = Query(None, description="Comma-separated list of detected types")):
    """
    Return the progress.json for a given repo. If not present, triggers section content generation (mocked for now).
    """
    types = [t.strip() for t in detected_types.split(",")] if detected_types else []
    from api.langgraph.wiki_structure import load_structure_from_disk, get_wiki_structure
    structure = load_structure_from_disk(repo_url)
    if not structure:
        structure = get_wiki_structure(repo_url, types, use_llm=True)
    # Use new directory structure
    repo_dir = get_repo_data_dir(repo_url)
    progress_path = os.path.join(repo_dir, "progress.json")
    if not os.path.exists(progress_path):
        # Trigger generation (mocked)
        generate_section_content(repo_url, types)
    if not os.path.exists(progress_path):
        return {"error": "Progress not found"}, HTTP_404_NOT_FOUND
    with open(progress_path) as f:
        return json.load(f)

@app.get("/wiki-section")
async def wiki_section(repo_url: str = Query(..., description="Repository URL or path"), section_id: str = Query(..., description="Section ID"), detected_types: Optional[str] = Query(None, description="Comma-separated list of detected types")):
    """
    Return the Markdown content for a given section. 404 if not found.
    """
    import os
    from api.langgraph.wiki_structure import get_repo_data_dir
    
    # Use new directory structure
    repo_dir = get_repo_data_dir(repo_url)
    md_path = os.path.join(repo_dir, f"{section_id}.md")
    
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
    
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
        
    def pipeline():
        try:
            # Step 1: Clone/copy repo if needed
            repo_is_git = repo_url.startswith("http://") or repo_url.startswith("https://") or repo_url.startswith("git@")
            repo_dest = os.path.join(repos_dir, repo_id)
            wiki_data_dir = os.path.join("wiki-data", "wikis", repo_id)
            if not os.path.exists(repo_dest):
                progress["status"] = "cloning/copying repository"
                progress["log"].append(f"{'Cloning' if repo_is_git else 'Copying'} repository to {repo_dest}...")
                with open(progress_path, "w") as f:
                    json.dump(progress, f, indent=2)
                if repo_is_git:
                    # Clone repo
                    try:
                        subprocess.run(["git", "clone", repo_url, repo_dest], check=True)
                    except Exception as e:
                        progress["status"] = "error"
                        progress["log"].append(f"Error cloning repo: {e}")
                        with open(progress_path, "w") as f:
                            json.dump(progress, f, indent=2)
                        return progress
                else:
                    shutil.copytree(repo_url, repo_dest)
            # --- Clean up wiki data before reindexing ---
            if os.path.exists(wiki_data_dir):
                try:
                    shutil.rmtree(wiki_data_dir)
                    progress["log"].append(f"Deleted existing wiki data for {repo_id} before reindexing.")
                except Exception as e:
                    progress["log"].append(f"Error deleting wiki data for {repo_id}: {e}")
            os.makedirs(wiki_data_dir, exist_ok=True)
            # --- RAG pipeline trigger follows ---
            progress["status"] = "indexing (RAG pipeline)"
            progress["log"].append("Running RAG pipeline (embedding/indexing)...")
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
            try:
                run_rag_pipeline(repo_id, embedding_provider="ollama_nomic", skip_indexing=False)
            except Exception as e:
                progress["status"] = "error"
                progress["log"].append(f"Error running RAG pipeline: {e}")
                with open(progress_path, "w") as f:
                    json.dump(progress, f, indent=2)
                return progress

            progress["status"] = "scanning repository"
            progress["log"].append("Scanning repository...")
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
                
            # Step 2: Generate wiki structure (using specified model)
            model_desc = f"{model_name} (temp={temperature}, top_p={top_p})" if model != "deterministic" else "deterministic"
            progress["status"] = f"generating wiki structure ({model_desc})"
            progress["log"].append(f"Generating wiki structure using {model_desc}...")
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
            
            # Verify the repository path exists
            if not os.path.exists(repo_dest):
                progress["status"] = "error"
                progress["log"].append(f"Error: Repository path not found at {repo_dest}")
                with open(progress_path, "w") as f:
                    json.dump(progress, f, indent=2)
                return
            
            # Determine if we should use LLM based on model parameter
            use_llm = model != "deterministic"
            use_openai = model == "openai"
            
            # Create generation_config to pass to the structure generation function
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "model_name": model_name
            }
            
            try:
                # Store the original repo_url in the structure, but use repo_dest for context gathering
                structure = get_wiki_structure(
                    repo_url,  # Use original repo_url for identification 
                    detected_types=types, 
                    use_llm=use_llm,
                    use_openai=use_openai if use_llm else False,
                    generation_config=generation_config,
                    repo_context_path=repo_dest  # Pass the actual repository path for context
                )
                
                progress["status"] = "wiki structure generated"
                progress["log"].append(f"Wiki structure generated using {model_desc}.")
                with open(progress_path, "w") as f:
                    json.dump(progress, f, indent=2)
            except Exception as e:
                progress["status"] = "error"
                progress["log"].append(f"Error generating wiki structure: {str(e)}")
                with open(progress_path, "w") as f:
                    json.dump(progress, f, indent=2)
                return
                
            # Step 3: Generate section content
            progress["status"] = "generating section content"
            progress["log"].append("Generating section content...")
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
            generate_section_content(repo_url, types)
            progress["status"] = "done"
            progress["log"].append("Wiki generation complete.")
            progress["finished_at"] = datetime.utcnow().isoformat() + "Z"
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            progress["status"] = "error"
            progress["log"].append(f"Error: {str(e)}")
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
                
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
