"""
LangGraph-based RAG pipeline scaffold for DeepWiki

- OpenAI for embeddings
- Gemini for LLM
- ChromaDB for vector storage (to be integrated)

This file provides the initial graph structure and node stubs.
"""

from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, List, Set, Callable, Tuple, Optional
import os
import re
from langchain_community.document_loaders import GitLoader, DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# Use the community version until langchain_chroma is installed
from langchain_community.vectorstores import Chroma
import hashlib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime
from api.langgraph_config import config
import time

# --- Core State Type ---
class RAGState(Dict[str, Any]):
    """State object passed between nodes in the LangGraph pipeline."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGState":
        """Create a RAGState from a dictionary."""
        state = cls()
        for key, value in data.items():
            state[key] = value
        return state
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return dict(self)

# --- File Inclusion/Exclusion Filters ---
def get_directory_exclusion_patterns() -> List[str]:
    """Returns patterns for directories to exclude"""
    return [
        # Common build and dependency directories
        'node_modules',
        'bower_components',
        'vendor',
        'build',
        'dist',
        'target',
        'out',
        '__pycache__',  # Python cache directories
        '.git',
        '.github',
        '.svn',
        '.idea',
        '.vscode',
        '.gradle',
        '.cache',
        
        # Virtual environments
        'venv',
        'virtualenv',
        '.env',
        'env',
        '.venv',
        
        # Generated documentation
        'docs/build',
        'site',
        
        # Mobile specific
        'Pods',
        
        # Logs and temporary files
        'logs',
        'tmp',
        'temp',
    ]

def get_file_extension_inclusions() -> Set[str]:
    """Returns file extensions to include"""
    return {
        # Programming languages
        '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.cpp', '.c', '.h', '.cs', '.rb', '.php', '.swift', '.kt', '.rs', '.scala',
        
        # Configuration
        '.json', '.yaml', '.yml', '.toml', '.ini', '.xml', '.env.example',
        
        # Web
        '.html', '.css', '.scss', '.sass', '.less',
        
        # Documentation
        '.md', '.markdown', '.rst', '.txt',
        
        # Shell scripts
        '.sh', '.bash', '.zsh', '.bat', '.ps1',
    }

def get_binary_extensions_exclusions() -> Set[str]:
    """Returns extensions of binary files to exclude"""
    return {
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.svg', '.webp',
        
        # Audio/Video
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.webm', '.ogg', '.flac',
        
        # Archives
        '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz',
        
        # Executables and libraries
        '.exe', '.dll', '.so', '.dylib', '.jar', '.war', '.ear', '.whl', '.pyc', '.class',
        
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        
        # Database and data
        '.db', '.sqlite', '.sqlite3', '.mdb', '.csv',
        
        # Mobile specific
        '.apk', '.ipa',
        
        # Misc
        '.ttf', '.otf', '.woff', '.woff2', '.eot',
    }

def should_include_file(file_path: str) -> bool:
    """Determines if a file should be included in document processing"""
    # Skip files in excluded directories or cache files
    if '__pycache__' in file_path:
        return False
        
    for pattern in get_directory_exclusion_patterns():
        if pattern in file_path.split(os.sep):
            return False
    
    # Check file extension
    _, ext = os.path.splitext(file_path.lower())
    
    # Skip binary files
    if ext in get_binary_extensions_exclusions():
        return False
    
    # Skip Python cache files (.pyc, .pyo)
    if ext in {'.pyc', '.pyo'}:
        return False
    
    # Include only files with extensions we care about
    if ext not in get_file_extension_inclusions():
        return False
    
    # Skip files that are too large (more than 1MB)
    try:
        if os.path.getsize(file_path) > 1024 * 1024:
            return False
    except (OSError, IOError):
        # If we can't check the size, skip it
        return False
        
    return True

def custom_file_filter(file_path: str) -> bool:
    """Filter function for GitLoader"""
    # Skip Python cache files/directories
    if '__pycache__' in file_path:
        return False
        
    # For GitLoader, we don't have the full path, just the file path within the repo
    # Skip files in excluded directories
    for pattern in get_directory_exclusion_patterns():
        if pattern in file_path.split('/'):  # GitLoader uses forward slashes
            return False
    
    # Check file extension
    _, ext = os.path.splitext(file_path.lower())
    
    # Skip binary files
    if ext in get_binary_extensions_exclusions():
        return False
    
    # Skip Python cache files (.pyc, .pyo)
    if ext in {'.pyc', '.pyo'}:
        return False
        
    # Include only files with extensions we care about
    if ext not in get_file_extension_inclusions():
        return False
        
    return True

# --- Node Implementations ---
def load_documents_node(state: RAGState) -> RAGState:
    """
    Loads documents from a Git repository (URL) or a local directory path.
    Expects 'repo_identifier' in state (either a URL or a local path).
    Stores a list of langchain Document objects in state['documents'].
    """
    # Access the repository identifier from the state dictionary directly
    repo_identifier = state.get("repo_identifier")
    
    # Print some debug information
    print(f"State keys in load_documents_node: {state.keys()}")
    print(f"Repository identifier: {repo_identifier}")
    
    if not repo_identifier:
        raise ValueError("No 'repo_identifier' provided in state.")

    # Normalize repository identifier
    if isinstance(repo_identifier, str):
        # Remove trailing slashes for consistency
        repo_identifier = repo_identifier.rstrip("/")
        # Expand user directory (e.g., ~/Dev to /home/user/Dev)
        if repo_identifier.startswith("~"):
            repo_identifier = os.path.expanduser(repo_identifier)
        # Convert to absolute path if it's a local path
        if not (repo_identifier.startswith("http://") or repo_identifier.startswith("https://")):
            repo_identifier = os.path.abspath(repo_identifier)
    
    # Heuristic: treat as URL if it starts with http(s)://, else as local path
    if repo_identifier.startswith("http://") or repo_identifier.startswith("https://"):
        # Git repository
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            try:
                # Try to determine the default branch
                import subprocess
                possible_branches = []
                try:
                    # Try to get the default branch name
                    cmd = ["git", "ls-remote", "--symref", repo_identifier, "HEAD"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and "ref: refs/heads/" in result.stdout:
                        for line in result.stdout.splitlines():
                            if line.startswith("ref: refs/heads/"):
                                branch_line = line.split()
                                # The branch name is after 'refs/heads/'
                                branch = branch_line[1].split("refs/heads/")[-1].strip()
                                if branch:
                                    possible_branches.append(branch)
                                    break
                except (subprocess.SubprocessError, IndexError, TimeoutError):
                    pass
                # Always try 'main' and 'master' as fallbacks
                for fallback in ["main", "master"]:
                    if fallback not in possible_branches:
                        possible_branches.append(fallback)
                # Try each branch until one works
                last_error = None
                for branch in possible_branches:
                    try:
                        loader = GitLoader(
                            clone_url=repo_identifier,
                            repo_path=tmpdir,
                            file_filter=custom_file_filter,
                            branch=branch
                        )
                        documents = loader.load()
                        print(f"Loaded {len(documents)} documents from Git repository {repo_identifier} (branch: {branch})")
                        break
                    except Exception as e:
                        print(f"Branch '{branch}' failed: {e}")
                        last_error = e
                        documents = None
                if documents is None:
                    raise ValueError(f"Failed to load Git repository: {last_error}")
            except Exception as e:
                print(f"Error loading Git repository: {str(e)}")
                raise ValueError(f"Failed to load Git repository: {str(e)}")
    else:
        # Local directory
        if not os.path.exists(repo_identifier) or not os.path.isdir(repo_identifier):
            raise ValueError(f"Local path does not exist or is not a directory: {repo_identifier}")
        
        print(f"Loading documents from local directory: {repo_identifier}")
        
        # DirectoryLoader interface changed over time - try different approaches
        try:
            # Create a more compatible filter function
            def is_valid_file(file_path):
                return should_include_file(file_path)
            
            # Track skipped files for summary reporting
            skipped_files = {
                'cache_files': 0,
                'excluded_dirs': 0,
                'excluded_extensions': 0,
                'too_large': 0,
                'other': 0
            }
            
            def count_skipped_file(file_path):
                if '__pycache__' in file_path or file_path.endswith('.pyc') or file_path.endswith('.pyo'):
                    skipped_files['cache_files'] += 1
                    return False
                
                # Check excluded directories
                for pattern in get_directory_exclusion_patterns():
                    if pattern in file_path.split(os.sep):
                        skipped_files['excluded_dirs'] += 1
                        return False
                
                # Check file extension
                _, ext = os.path.splitext(file_path.lower())
                
                # Skip binary files
                if ext in get_binary_extensions_exclusions():
                    skipped_files['excluded_extensions'] += 1
                    return False
                
                # Include only files with extensions we care about
                if ext not in get_file_extension_inclusions():
                    skipped_files['excluded_extensions'] += 1
                    return False
                
                # Skip files that are too large (more than 1MB)
                try:
                    if os.path.getsize(file_path) > 1024 * 1024:
                        skipped_files['too_large'] += 1
                        return False
                except (OSError, IOError):
                    # If we can't check the size, skip it
                    skipped_files['other'] += 1
                    return False
                    
                return True
            
            # First attempt: Try with modern parameters (load_filters)
            try:
                print("Trying DirectoryLoader with load_filters parameter")
                loader = DirectoryLoader(
                    repo_identifier,
                    glob="**/*.*",
                    loader_cls=TextLoader,
                    show_progress=True,
                    use_multithreading=True,
                    silent_errors=True,
                    load_filters=[count_skipped_file]
                )
                documents = loader.load()
                print(f"Successfully loaded with load_filters: {len(documents)} documents")
            except (TypeError, AttributeError) as e1:
                # Second attempt: Try with filter parameter
                print(f"First attempt failed with error: {str(e1)}")
                print("Trying DirectoryLoader with filter parameter")
                loader = DirectoryLoader(
                    repo_identifier,
                    glob="**/*.*",
                    loader_cls=TextLoader,
                    show_progress=True,
                    use_multithreading=True,
                    silent_errors=True,
                    filter=count_skipped_file
                )
                documents = loader.load()
                print(f"Successfully loaded with filter: {len(documents)} documents")
        except Exception as e:
            # Fallback to minimal parameters if all else fails
            print(f"Standard approaches failed with error: {str(e)}")
            print("Falling back to minimal DirectoryLoader parameters")
            try:
                skipped_files = {'total': 0}
                loader = DirectoryLoader(
                    repo_identifier,
                    glob="**/*.*",
                    loader_cls=TextLoader,
                    silent_errors=True
                )
                raw_documents = loader.load()
                
                # Manually filter documents since we couldn't use the loader's filter
                documents = []
                for doc in raw_documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source_path = doc.metadata.get('source', '')
                        if should_include_file(source_path):
                            documents.append(doc)
                        else:
                            skipped_files['total'] += 1
                
                print(f"Loaded and manually filtered to {len(documents)} documents from {len(raw_documents)} total")
            except Exception as e_final:
                print(f"All loading approaches failed. Final error: {str(e_final)}")
                raise ValueError(f"Unable to load documents from {repo_identifier}: {str(e_final)}")

        # Print summary of skipped files
        if 'total' in skipped_files:
            print(f"Skipped {skipped_files['total']} files in total")
        else:
            print(f"Skipped files summary:")
            print(f"  - Cache files (__pycache__, .pyc, etc.): {skipped_files['cache_files']}")
            print(f"  - Files in excluded directories: {skipped_files['excluded_dirs']}")
            print(f"  - Files with excluded extensions: {skipped_files['excluded_extensions']}")
            print(f"  - Files too large (>1MB): {skipped_files['too_large']}")
            print(f"  - Other skipped files: {skipped_files['other']}")

    # Ensure all documents have solid metadata
    for doc in documents:
        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
            doc.metadata = {}
        # Add file_path and type if missing
        if 'file_path' not in doc.metadata:
            doc.metadata['file_path'] = doc.metadata.get('source', 'unknown')
        if 'type' not in doc.metadata:
            ext = os.path.splitext(doc.metadata['file_path'])[1].lower()
            doc.metadata['type'] = ext.lstrip('.')
        # Add repository identifier to allow multi-repo queries later
        doc.metadata['repository_id'] = repo_identifier

    state["documents"] = documents
    print(f"Loaded {len(documents)} documents")
    return state


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
                separators=["\nclass ", "\ndef ", "\nfunction ", "\n# ", "\n", " ", ""],
            )
        elif ext in markdown_exts:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n## ", "\n# ", "\n", " ", ""],
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n", " ", ""],
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


def embed_documents_node(state: RAGState) -> RAGState:
    """
    Embeds all chunks using OpenAI embeddings. Stores embeddings in state['embeddings'] and updates chunk metadata.
    """
    chunks = state.get('chunks', [])
    if not chunks:
        raise ValueError("No chunks found in state for embedding.")

    embedder = OpenAIEmbeddings()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.embed_documents(texts)

    # Attach embedding to each chunk's metadata
    for chunk, emb in zip(chunks, embeddings):
        chunk.metadata['embedding'] = emb

    state['embeddings'] = embeddings
    state['chunks'] = chunks  # Now with embeddings in metadata
    return state

def generate_collection_name(repo_identifier: str) -> str:
    """
    Generates a safe collection name from a repository identifier.
    For URLs: Uses the last part of the URL (repo name)
    For local paths: Uses a hash of the absolute path
    
    Returns a unique, valid collection name for ChromaDB.
    """
    # Create a prefix to categorize the collection type
    if repo_identifier.startswith("http://") or repo_identifier.startswith("https://"):
        # For URLs: Extract the repository name
        # Remove .git suffix if present
        path_parts = repo_identifier.rstrip("/").split("/")
        
        # Try to extract owner and repo name if possible
        if len(path_parts) >= 2:
            owner = path_parts[-2]
            repo = path_parts[-1].replace(".git", "")
            prefix = f"git_{owner}_{repo}"
        else:
            # Fallback if we can't parse the URL as expected
            repo_name = path_parts[-1].replace(".git", "")
            prefix = f"git_{repo_name}"
            
        # Remove any special characters that could cause issues in ChromaDB collection names
        prefix = re.sub(r'[^\w\-_]', '_', prefix)
        
        # Ensure the collection name is not too long for ChromaDB (max is usually around 64 chars)
        if len(prefix) > 60:
            # Take the first 50 chars and add a hash of the full name
            short_hash = hashlib.md5(repo_identifier.encode()).hexdigest()[:10]
            prefix = f"{prefix[:50]}_{short_hash}"
    else:
        # For local paths: Hash the absolute path for a consistent identifier
        abs_path = os.path.abspath(repo_identifier)
        dir_name = os.path.basename(abs_path)
        # Clean the directory name
        dir_name = re.sub(r'[^\w\-_]', '_', dir_name)
        # Add a hash of the full path to ensure uniqueness
        path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:10]
        prefix = f"local_{dir_name}_{path_hash}"
    
    return prefix

def get_persistent_dir() -> str:
    """Returns the directory for persistent ChromaDB storage"""
    # Use home directory with a hidden folder (similar to the .adalflow approach)
    home_dir = os.path.expanduser("~")
    chroma_dir = os.path.join(home_dir, ".deepwiki", "chromadb")
    os.makedirs(chroma_dir, exist_ok=True)
    return chroma_dir

def store_vectors_node(state: RAGState) -> RAGState:
    """
    Stores document chunks and their embeddings in a persistent ChromaDB collection.
    Each repository gets its own collection.
    """
    # Import config within the function to avoid global reference issues
    try:
        from api.langgraph_config import config
    except ImportError:
        # Fallback if import fails
        from dataclasses import dataclass
        
        @dataclass
        class DefaultConfig:
            update_existing_collections = False
        
        config = DefaultConfig()
    
    chunks = state.get('chunks', [])
    embeddings = state.get('embeddings', [])
    repo_identifier = state.get('repo_identifier')
    
    if not chunks or not embeddings or not repo_identifier:
        raise ValueError("Missing required data (chunks, embeddings, or repo_identifier) for vector storage.")
    
    # Generate a collection name based on the repository
    collection_name = generate_collection_name(repo_identifier)
    persistent_dir = get_persistent_dir()
    
    print(f"Storing vectors in ChromaDB collection '{collection_name}' at {persistent_dir}")
    
    # Initialize embedding function to match how we embed
    embedding_function = OpenAIEmbeddings()
    
    # Maximum number of retries for collection handling
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create a ChromaDB client
            import chromadb
            import time
            import shutil
            import os
            import random
            
            # Get client
            client = get_chroma_client(persistent_dir)
            force_recreate = False
            
            # First try to check if collection exists
            try:
                collections = client.list_collections()
                exists = any(c.name == collection_name for c in collections)
            except KeyError as e:
                # Handle '_type' KeyError which indicates a ChromaDB format issue
                if "'_type'" in str(e):
                    print(f"ChromaDB format error detected: {e}")
                    print("Will attempt to recreate collection...")
                    force_recreate = True
                    exists = False
                else:
                    raise e
            except Exception as e:
                print(f"Error listing collections: {e}")
                print("Will attempt to recreate collection...")
                force_recreate = True
                exists = False
            
            # Check if recreate collection
            recreate = force_recreate or not config.update_existing_collections
            
            # Delete if needed
            if exists and recreate:
                try:
                    print(f"Deleting existing collection '{collection_name}'")
                    # Wait for a random time to avoid race conditions with multiple processes
                    time.sleep(random.uniform(0.1, 0.5))
                    
                    # Try to delete collection
                    client.delete_collection(collection_name)
                    
                    # Verify it was deleted successfully
                    time.sleep(0.5)  # Wait to ensure deletion completes
                    collections = client.list_collections()
                    if any(c.name == collection_name for c in collections):
                        print(f"Warning: Collection '{collection_name}' still exists after deletion attempt")
                        # Force a client recreation
                        client = None
                    else:
                        exists = False
                except Exception as del_error:
                    print(f"Warning: Failed to delete collection: {del_error}")
                    # We'll try to continue anyway by recreating the client
                    client = None
            
            # If client deletion failed, recreate the client
            if client is None:
                print("Recreating ChromaDB client...")
                time.sleep(0.5)  # Wait before recreating client
                # Try to delete any lock files
                lock_file = os.path.join(persistent_dir, ".lock")
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                        print(f"Removed lock file before client recreation: {lock_file}")
                    except:
                        pass
                client = get_chroma_client(persistent_dir)
                exists = False
            
            # Get or create the collection
            chroma_collection = None
            
            # Use try/except to handle "Collection already exists" errors
            try:
                if exists and not force_recreate:
                    print(f"Using existing collection '{collection_name}'")
                    try:
                        chroma_collection = client.get_collection(name=collection_name)
                    except Exception as e:
                        print(f"Error accessing existing collection: {e}")
                        # Try to recreate instead
                        exists = False
                
                if not exists or force_recreate:
                    print(f"Creating new collection '{collection_name}'")
                    try:
                        chroma_collection = client.create_collection(name=collection_name)
                    except chromadb.errors.UniqueConstraintError:
                        print(f"Collection '{collection_name}' already exists during creation")
                        # Collection exists but we couldn't see it earlier
                        # Try to get it instead
                        time.sleep(0.5)  # Wait before retry
                        chroma_collection = client.get_collection(name=collection_name)
                    except Exception as create_error:
                        raise ValueError(f"Failed to create collection: {create_error}")
            
            except Exception as coll_error:
                print(f"Error with collection: {coll_error}")
                
                # If this is our last retry, try a full reset
                if retry_count == max_retries - 1:
                    print("Attempting to recreate ChromaDB from scratch...")
                    
                    # Create a unique backup directory
                    backup_dir = f"{persistent_dir}_backup_{int(time.time())}_{random.randint(1000, 9999)}"
                    try:
                        # Safely delete the ChromaDB directory and recreate it
                        if os.path.exists(persistent_dir):
                            # Backup first
                            shutil.copytree(persistent_dir, backup_dir)
                            print(f"Created ChromaDB backup at {backup_dir}")
                            
                            # Remove lock files first
                            lock_file = os.path.join(persistent_dir, ".lock")
                            if os.path.exists(lock_file):
                                os.remove(lock_file)
                                print(f"Removed lock file: {lock_file}")
                                
                            # Instead of removing the entire directory, rename it to avoid race conditions
                            temp_dir = f"{persistent_dir}_old_{int(time())}"
                            os.rename(persistent_dir, temp_dir)
                            print(f"Renamed old ChromaDB directory to {temp_dir}")
                            
                            # Create fresh directory
                            os.makedirs(persistent_dir, exist_ok=True)
                            
                            # Create a new client with the fresh directory
                            client = chromadb.PersistentClient(path=persistent_dir)
                            
                            # Create a fresh collection
                            chroma_collection = client.create_collection(name=collection_name)
                            print(f"Successfully recreated ChromaDB collection '{collection_name}'")
                            
                            # Now it's safe to remove the old directory
                            try:
                                shutil.rmtree(temp_dir)
                            except:
                                pass  # Ignore errors in cleanup
                    except Exception as rebuild_error:
                        # If even this fails, we give up
                        print(f"Failed to rebuild ChromaDB: {rebuild_error}")
                        raise ValueError(f"Unable to rebuild ChromaDB after multiple attempts: {rebuild_error}")
                else:
                    # Increment retry count and try again
                    retry_count += 1
                    print(f"Retrying collection creation (attempt {retry_count}/{max_retries})...")
                    time.sleep(1)  # Wait before retry
                    continue
            
            # If we got here, we have a valid collection, so create the vectorstore
            vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_function
            )
            
            # Extract document content, metadata, and embeddings
            doc_embeddings = [chunk.metadata.pop('embedding', None) for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            texts = [chunk.page_content for chunk in chunks]
            ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
            
            # Add documents with their embeddings to ChromaDB
            try:
                # Use the ChromaDB collection directly for better control
                chroma_collection.add(
                    embeddings=doc_embeddings,
                    metadatas=metadatas,
                    documents=texts,
                    ids=ids
                )
                print(f"Successfully stored {len(chunks)} documents in ChromaDB collection '{collection_name}'")
                
                # Add vectorstore to state for retrieval to use later
                state['vectorstore'] = vectorstore
                state['collection_name'] = collection_name
                
                # We succeeded, so break out of the retry loop
                break
                
            except Exception as e:
                print(f"Error storing documents in ChromaDB: {e}")
                
                # For the last retry, fail completely
                if retry_count == max_retries - 1:
                    raise ValueError(f"Failed to store vectors after multiple attempts: {e}")
                
                # Otherwise increment retry count and try again
                retry_count += 1
                print(f"Retrying vector storage (attempt {retry_count}/{max_retries})...")
                time.sleep(1)  # Wait before retry
                
        except Exception as e:
            print(f"Error in store_vectors_node: {e}")
            
            # For the last retry, fail completely
            if retry_count == max_retries - 1:
                raise ValueError(f"Failed to initialize ChromaDB after multiple attempts: {e}")
            
            # Otherwise increment retry count and try again
            retry_count += 1
            print(f"Retrying entire process (attempt {retry_count}/{max_retries})...")
            time.sleep(1)  # Wait before retry
    
    return state

def retrieve_node(state: RAGState) -> RAGState:
    """
    Retrieves relevant documents from ChromaDB based on a query.
    
    Expects:
    - state['query']: The search query
    - state['vectorstore']: The ChromaDB vectorstore initialized in store_vectors_node
    - state['collection_name']: The collection name for the current repository
    
    Returns:
    - state['relevant_documents']: List of retrieved documents
    """
    # Import config inside the function to avoid reference issues
    try:
        from api.langgraph_config import config
    except ImportError:
        # Fallback in case import fails
        from dataclasses import dataclass
        
        @dataclass
        class DefaultConfig:
            class Retriever:
                top_k = 20
            retriever = Retriever()
        
        config = DefaultConfig()
    
    query = state.get('query')
    vectorstore = state.get('vectorstore')
    collection_name = state.get('collection_name')
    repositories = state.get('repositories', [])  # Optional list of repositories to search
    repo_identifier = state.get('repo_identifier')
    
    if not query:
        raise ValueError("No 'query' provided in state for retrieval.")
    
    # Get the number of results to retrieve
    k = state.get('top_k', config.retriever.top_k)
    
    # Initialize lists to store retrieved documents and scores
    retrieved_docs = []
    
    # Case 1: Use the existing vectorstore if available
    if vectorstore:
        print(f"Using existing vectorstore for collection '{collection_name}'")
        try:
            retrieved_docs = vectorstore.similarity_search(query, k=k)
            print(f"Retrieved {len(retrieved_docs)} documents from collection '{collection_name}'")
        except Exception as e:
            print(f"Error retrieving from existing vectorstore: {str(e)}")
            vectorstore = None  # Force regeneration
    
    # Case 2: Load vectorstore for the primary repository if not already available
    if not vectorstore and repo_identifier:
        try:
            print(f"Loading vectorstore for repository: {repo_identifier}")
            collection_name = generate_collection_name(repo_identifier)
            persistent_dir = get_persistent_dir()
            
            # Import chromadb directly
            import chromadb
            import time
            
            # Initialize ChromaDB with the persistent directory and collection
            try:
                # Create a client
                client = chromadb.PersistentClient(path=persistent_dir)
                
                # Get or create the collection
                try:
                    # Check if collection exists
                    collections = []
                    try:
                        collections = client.list_collections()
                    except KeyError as e:
                        # Handle '_type' KeyError which indicates a ChromaDB format issue
                        if "'_type'" in str(e):
                            print(f"ChromaDB format error detected: {e}")
                            print("Cannot retrieve from corrupted database. Please reindex the repository.")
                            state['relevant_documents'] = []
                            return state
                        else:
                            raise e
                    except Exception as e:
                        print(f"Error listing collections: {e}")
                        state['relevant_documents'] = []
                        return state
                    
                    exists = any(c.name == collection_name for c in collections)
                    
                    if exists:
                        try:
                            print(f"Collection '{collection_name}' exists, loading...")
                            chroma_collection = client.get_collection(name=collection_name)
                            
                            # Create the LangChain wrapper for the collection
                            vectorstore = Chroma(
                                client=client,
                                collection_name=collection_name,
                                embedding_function=OpenAIEmbeddings()
                            )
                            
                            # Add to state
                            state['vectorstore'] = vectorstore
                            state['collection_name'] = collection_name
                            
                            # Perform the similarity search
                            retrieved_docs = vectorstore.similarity_search(query, k=k)
                            print(f"Retrieved {len(retrieved_docs)} documents from collection '{collection_name}'")
                        except Exception as e:
                            print(f"Error accessing collection: {e}")
                            # Skip this collection and return empty results
                            state['relevant_documents'] = []
                            return state
                    else:
                        print(f"Collection '{collection_name}' not found, cannot retrieve documents.")
                        # Return empty results
                        state['relevant_documents'] = []
                        return state
                except Exception as e:
                    print(f"Error checking collection existence: {e}")
                    # Return empty results
                    state['relevant_documents'] = []
                    return state
                
            except Exception as e:
                print(f"Error initializing ChromaDB: {e}")
                # Return empty results
                state['relevant_documents'] = []
                return state
                
        except Exception as e:
            print(f"Error loading vectorstore for {repo_identifier}: {str(e)}")
            # Don't fail the query if the vectorstore isn't available
            vectorstore = None
            retrieved_docs = []
    
    # Case 3: Handle multi-repository search if requested
    if repositories and len(repositories) > 0:
        print(f"Performing multi-repository search across {len(repositories)} repositories")
        multi_repo_docs = []
        
        import chromadb
        
        for repo in repositories:
            try:
                # Skip if it's the same as the primary repository (we already searched it)
                if repo == repo_identifier:
                    continue
                    
                repo_collection_name = generate_collection_name(repo)
                print(f"Searching repository: {repo} (collection: {repo_collection_name})")
                
                try:
                    # Create a client
                    client = chromadb.PersistentClient(path=get_persistent_dir())
                    
                    # Check if collection exists
                    try:
                        collections = client.list_collections()
                        exists = any(c.name == repo_collection_name for c in collections)
                    except KeyError as e:
                        # Handle '_type' KeyError which indicates a ChromaDB format issue
                        if "'_type'" in str(e):
                            print(f"ChromaDB format error detected for {repo}: {e}")
                            print(f"Skipping repository {repo}")
                            continue
                        else:
                            raise e
                    except Exception as e:
                        print(f"Error listing collections for {repo}: {e}")
                        continue
                    
                    if not exists:
                        print(f"Collection '{repo_collection_name}' not found, skipping...")
                        continue
                    
                    # Initialize a new vectorstore for this repository
                    try:
                        repo_vectorstore = Chroma(
                            client=client,
                            collection_name=repo_collection_name,
                            embedding_function=OpenAIEmbeddings()
                        )
                        
                        # Perform search for this repository
                        repo_docs = repo_vectorstore.similarity_search(query, k=k//2)  # Use smaller k for each repo
                        print(f"Retrieved {len(repo_docs)} documents from {repo}")
                        
                        # Add repository information to the metadata
                        for doc in repo_docs:
                            doc.metadata['repository'] = repo
                        
                        multi_repo_docs.extend(repo_docs)
                    except Exception as e:
                        print(f"Error searching repository {repo}: {e}")
                        continue
                except Exception as e:
                    print(f"Error initializing ChromaDB for {repo}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error searching repository {repo}: {str(e)}")
        
        # Add multi-repository results to our list
        retrieved_docs.extend(multi_repo_docs)
        
        # Re-rank or truncate the combined results if needed
        if len(retrieved_docs) > k:
            # This is a simple truncation - in the future, we could implement a more sophisticated re-ranking
            retrieved_docs = retrieved_docs[:k]
    
    # Add the retrieved documents to the state
    state['relevant_documents'] = retrieved_docs
    
    # Log how many documents were retrieved
    print(f"Retrieved a total of {len(retrieved_docs)} documents")
    
    return state

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
    # Import dependencies within the function to avoid global import issues
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Import config inside the function to avoid reference issues
    try:
        from api.langgraph_config import config
    except ImportError:
        # Fallback in case relative import fails
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from api.langgraph_config import config
        except ImportError as e:
            print(f"Error importing config: {e}")
            # Create minimal default config if imports fail
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
    
    # --- NEW: Include all retrieved documents in context, with a safeguard for huge contexts ---
    MAX_CONTEXT_TOKENS = 900_000  # Leave some headroom for prompt/query
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
            # Estimate tokens as len(content) // 4 (very rough, but safe for code/text)
            est_tokens = len(content) // 4
            if total_tokens + est_tokens > MAX_CONTEXT_TOKENS:
                truncated = True
                break
            context_pieces.append(f"Document {i+1} from {file_path} in {repo_name}:\n{content}")
            total_tokens += est_tokens
        context_text = "\n\n".join(context_pieces)
        if truncated:
            print(f"[WARNING] Context truncated to ~{MAX_CONTEXT_TOKENS} tokens. Not all documents included.")
    
    # Define the system prompt
    system_template = """You are an expert code analyst and software documentation assistant.
Answer the user's question based on the provided context from the codebase.
If the context doesn't contain the information needed, say so clearly rather than making up information.
For code-related questions, include relevant code snippets and explain them.
For architecture or design questions, provide clear and structured explanations.
Format your response using Markdown for readability.

Context from the codebase:
{context}
"""
    
    # Define the human message template
    human_template = "{query}"
    
    # Format the system prompt with the context
    system_message = system_template.format(context=context_text)
    
    # Set up the LLM based on use_ollama
    try:
        if use_ollama:
            try:
                from langchain_community.llms.ollama import Ollama
                
                # Use the Ollama model specified in config
                print(f"Using Ollama model: {config.generator_ollama.model}")
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
            # Use Google Gemini model
            print(f"Using Gemini model: {config.generator.model}")
            llm = ChatGoogleGenerativeAI(
                model=config.generator.model,
                temperature=config.generator.temperature,
                top_p=config.generator.top_p,
            )
        
        # Create a simple prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template),
        ])
        
        # Create a chain to generate the answer
        chain = prompt | llm | StrOutputParser()
        
        # Generate the answer
        answer = chain.invoke({"query": query})
        
        # Add the answer to the state
        state['answer'] = answer
        
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(error_msg)
        state['answer'] = f"I couldn't generate a proper answer due to an error: {str(e)}"
        state['error_generate'] = str(e)
    
    return state

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class DialogTurn:
    id: str
    user_message: Message
    assistant_message: Optional[Message] = None

class ConversationMemory:
    """Simple conversation management with a list of dialog turns."""
    
    def __init__(self):
        self.dialog_turns: List[DialogTurn] = []
    
    def add_user_message(self, content: str) -> str:
        """Add a user message and return the turn ID."""
        turn_id = str(uuid4())
        self.dialog_turns.append(
            DialogTurn(
                id=turn_id,
                user_message=Message(role="user", content=content)
            )
        )
        return turn_id
    
    def add_assistant_message(self, turn_id: str, content: str) -> bool:
        """Add an assistant message to an existing turn."""
        for turn in self.dialog_turns:
            if turn.id == turn_id:
                turn.assistant_message = Message(role="assistant", content=content)
                return True
        return False
    
    def add_dialog_turn(self, user_content: str, assistant_content: str) -> str:
        """Add a complete dialog turn with user and assistant messages."""
        turn_id = str(uuid4())
        self.dialog_turns.append(
            DialogTurn(
                id=turn_id,
                user_message=Message(role="user", content=user_content),
                assistant_message=Message(role="assistant", content=assistant_content)
            )
        )
        return turn_id
    
    def get_messages(self, limit: int = None) -> List[Tuple[str, str]]:
        """Get a list of message tuples (role, content) from the conversation."""
        messages = []
        for turn in self.dialog_turns[-limit:] if limit else self.dialog_turns:
            messages.append(("user", turn.user_message.content))
            if turn.assistant_message:
                messages.append(("assistant", turn.assistant_message.content))
        return messages
    
    def to_dict(self) -> Dict:
        """Convert the conversation to a dictionary for serialization."""
        return {
            "dialog_turns": [
                {
                    "id": turn.id,
                    "user_message": {
                        "role": turn.user_message.role,
                        "content": turn.user_message.content,
                        "timestamp": turn.user_message.timestamp.isoformat(),
                    },
                    "assistant_message": {
                        "role": turn.assistant_message.role,
                        "content": turn.assistant_message.content,
                        "timestamp": turn.assistant_message.timestamp.isoformat(),
                    } if turn.assistant_message else None,
                }
                for turn in self.dialog_turns
            ]
        }

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
    
    # Ensure we have a valid query
    if not query:
        error_msg = "Missing 'query' in state for memory."
        print(error_msg)
        state['error_memory'] = error_msg
        return state
    
    # If no answer was generated (due to an error), create a fallback response
    if not answer:
        error_message = state.get('error_generate', "An unknown error occurred")
        answer = f"I'm sorry, I couldn't generate a proper response. Error: {error_message}"
        state['answer'] = answer
        print(f"Created fallback answer for memory: {answer[:50]}...")
    
    # Get or initialize conversation memory
    memory = state.get('memory')
    if not memory or not isinstance(memory, ConversationMemory):
        memory = ConversationMemory()
        print("Created new ConversationMemory instance")
    
    # Add the current turn to memory
    try:
        memory.add_dialog_turn(user_content=query, assistant_content=answer)
        print(f"Added dialog turn to memory with query: {query[:30]}... and answer: {answer[:30]}...")
    except Exception as e:
        error_msg = f"Error adding dialog turn to memory: {str(e)}"
        print(error_msg)
        state['error_memory'] = error_msg
        return state
    
    # Update the state with the memory
    state['memory'] = memory
    print("Successfully updated conversation memory")
    
    return state

# --- Graph Construction ---
graph = StateGraph(RAGState)

graph.add_node("load_documents", load_documents_node)
graph.add_node("split_text", split_text_node)
graph.add_node("embed_documents", embed_documents_node)
graph.add_node("store_vectors", store_vectors_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("memory", memory_node)

# Add an edge from START to the first node
graph.add_edge(START, "load_documents")

# Example linear flow (to be refined as needed)
graph.add_edge("load_documents", "split_text")
graph.add_edge("split_text", "embed_documents")
graph.add_edge("embed_documents", "store_vectors")
graph.add_edge("store_vectors", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "memory")
graph.add_edge("memory", END)

# Export the graph for use in the API
rag_graph = graph.compile()

# --- Create a simplified test graph for debugging ---
def debug_rag_pipeline(state: RAGState) -> dict:
    """
    A simplified pipeline that doesn't use the graph structure for debugging.
    This function runs each node in sequence and provides detailed debugging
    information for troubleshooting.
    """
    # Import needed modules
    import traceback
    import time
    
    # Make a copy of the state to avoid modifying the input
    result_state = RAGState.from_dict(state.to_dict())
    
    print(f"\n==== DEBUG PIPELINE START ====")
    print(f"Input state: {result_state}")
    
    # Define the node sequence
    node_sequence = [
        ("load_documents", load_documents_node),
        ("split_text", split_text_node),
        ("embed_documents", embed_documents_node),
        ("store_vectors", store_vectors_node),
        ("retrieve", retrieve_node),
        ("generate", generate_node),
        ("memory", memory_node)
    ]
    
    # Run each node in sequence with verbose debugging
    for node_name, node_func in node_sequence:
        try:
            print(f"\n==== Running node: {node_name} ====")
            
            # Execute the node
            start_time = time.time()
            
            # Make sure we catch any exceptions inside the node execution
            try:
                result_state = node_func(result_state)
                elapsed_time = time.time() - start_time
                
                # Log key stats after node execution
                print(f"Node {node_name} completed in {elapsed_time:.2f} seconds")
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"\n==== ERROR in node {node_name} after {elapsed_time:.2f} seconds ====")
                print(f"Error: {str(e)}")
                traceback.print_exc()
                
                # Add error information to the state
                result_state[f"error_{node_name}"] = str(e)
                result_state[f"error_traceback_{node_name}"] = traceback.format_exc()
                
                # Depending on the node, we might want to continue or stop
                if node_name in ["load_documents", "split_text", "embed_documents", "store_vectors"]:
                    print(f"Critical node {node_name} failed. Pipeline cannot continue.")
                    break
                else:
                    print(f"Non-critical node {node_name} failed. Attempting to continue pipeline.")
                continue
            
            # Report node-specific stats
            if node_name == "load_documents" and "documents" in result_state:
                print(f"Loaded {len(result_state['documents'])} documents")
            elif node_name == "split_text" and "chunks" in result_state:
                print(f"Created {len(result_state['chunks'])} chunks")
            elif node_name == "embed_documents" and "embeddings" in result_state:
                print(f"Generated {len(result_state['embeddings'])} embeddings")
            elif node_name == "store_vectors" and "vectorstore" in result_state:
                print(f"Stored vectors in collection: {result_state.get('collection_name', 'unknown')}")
            elif node_name == "retrieve" and "relevant_documents" in result_state:
                print(f"Retrieved {len(result_state['relevant_documents'])} documents")
                for i, doc in enumerate(result_state['relevant_documents'][:3]):  # Show top 3
                    print(f"  Top result {i+1}: {doc.metadata.get('file_path', 'unknown')} ({len(doc.page_content)} chars)")
            elif node_name == "generate" and "answer" in result_state:
                answer_preview = result_state['answer'][:100] + "..." if len(result_state['answer']) > 100 else result_state['answer']
                print(f"Generated answer: {answer_preview}")
                
        except Exception as e:
            # This should not happen as we're already catching exceptions inside the inner try-except block
            print(f"\n==== UNEXPECTED ERROR in debug_rag_pipeline for node {node_name} ====")
            print(f"Error: {str(e)}")
            traceback.print_exc()
    
    print(f"\n==== DEBUG PIPELINE COMPLETE ====")
    
    return result_state

# Modify the main run_rag_pipeline function to use the debug pipeline for testing
def run_rag_pipeline(
    repo_identifier: str,
    query: str,
    use_ollama: bool = False,
    top_k: int = None,
    memory = None,
    debug: bool = True,  # Set to True for testing
    repositories: List[str] = None,  # List of repositories to search
    skip_indexing: bool = False  # Skip indexing if collection already exists
) -> dict:
    """
    Main entry point to run the RAG pipeline.
    
    Args:
        repo_identifier: Repository URL or local path
        query: User query
        use_ollama: Whether to use Ollama models instead of OpenAI/Gemini
        top_k: Optional override for number of documents to retrieve
        memory: Optional existing conversation memory
        debug: Whether to use the debug pipeline (default: True for easier troubleshooting)
        repositories: Optional list of repositories to search (for multi-repository queries)
        skip_indexing: Skip document loading, splitting, and embedding if collection exists
        
    Returns:
        Dictionary with results including answer, conversation memory, and metadata
    """
    from datetime import datetime
    import chromadb
    
    # Import config inside the function to avoid reference issues
    try:
        from api.langgraph_config import config
    except ImportError:
        # Fallback in case import fails
        from dataclasses import dataclass
        
        @dataclass
        class DefaultConfig:
            class Retriever:
                top_k = 20
            retriever = Retriever()
        
        config = DefaultConfig()
    
    if not repo_identifier:
        raise ValueError("Repository identifier (URL or local path) is required")
    
    # For local paths, ensure they exist
    if not (repo_identifier.startswith("http://") or repo_identifier.startswith("https://")):
        repo_identifier = os.path.abspath(os.path.expanduser(repo_identifier))
        if not os.path.exists(repo_identifier):
            raise ValueError(f"Local directory does not exist: {repo_identifier}")
    
    # Normalize repositories if provided
    if repositories:
        normalized_repos = []
        for repo in repositories:
            if not (repo.startswith("http://") or repo.startswith("https://")):
                repo = os.path.abspath(os.path.expanduser(repo))
            normalized_repos.append(repo)
        repositories = normalized_repos
    
    print(f"Processing query for repository: {repo_identifier}")
    if repositories:
        print(f"Multi-repository mode enabled with {len(repositories)} repositories")
    
    # Timestamp for tracking performance
    start_time = time.time()
    
    # Get default top_k if not provided
    default_top_k = getattr(config.retriever, 'top_k', 20) if hasattr(config, 'retriever') else 20
    
    # Check if we can skip indexing by checking if the collection already exists
    collection_exists = False
    if skip_indexing:
        collection_name = generate_collection_name(repo_identifier)
        persistent_dir = get_persistent_dir()
        
        try:
            client = chromadb.PersistentClient(path=persistent_dir)
            collections = client.list_collections()
            collection_exists = any(c.name == collection_name for c in collections)
            
            if collection_exists:
                print(f"Collection '{collection_name}' exists. Skipping document loading and embedding.")
            else:
                print(f"Collection '{collection_name}' not found. Will perform full indexing.")
                skip_indexing = False
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            print("Will perform full indexing to be safe.")
            skip_indexing = False
    
    # Create input state dictionary with appropriate keys based on whether we're skipping indexing
    input_data = {
        "repo_identifier": repo_identifier,
        "query": query,
        "top_k": top_k or default_top_k,
        "use_ollama": use_ollama,
    }
    
    if memory is not None:
        input_data["memory"] = memory
        
    if repositories is not None:
        input_data["repositories"] = repositories
    
    # Create RAGState from dictionary
    state = RAGState.from_dict(input_data)
    
    # Debug information
    print(f"Initial state: {state}")
    
    # Run the pipeline (either debug or graph)
    try:
        if debug:
            # Use the simplified debugging pipeline with optional skipping
            if skip_indexing:
                # Create a shortened node sequence that skips indexing
                final_state = run_retrieval_only_pipeline(state, collection_name)
            else:
                # Run the full pipeline
                final_state = debug_rag_pipeline(state)
        else:
            # For the compiled graph, we can't easily skip nodes, so use the full graph
            final_state = rag_graph.invoke(state)
        
        print(f"Final state keys: {final_state.keys()}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return standardized response with results and metadata
        response = {
            # Primary output
            "answer": final_state.get("answer", ""),
            "memory": final_state.get("memory"),
            "retrieved_documents": final_state.get("relevant_documents", []),
            
            # Performance metrics and metadata
            "metadata": {
                "elapsed_time": elapsed_time,
                "repo_identifier": repo_identifier,
                "collection_name": final_state.get("collection_name"),
                "model_used": "ollama" if use_ollama else "openai/gemini",
                "top_k": top_k or default_top_k,
                "timestamp": datetime.now().isoformat(),
                "document_count": len(final_state.get("documents", [])) if "documents" in final_state else None,
                "chunk_count": len(final_state.get("chunks", [])) if "chunks" in final_state else None,
                "multi_repo_mode": bool(repositories),
                "repository_count": len(repositories) if repositories else 1,
                "skipped_indexing": skip_indexing,
            }
        }
        
        # Check for any errors that may have occurred in the debug pipeline
        errors = {k: v for k, v in final_state.items() if k.startswith("error_")}
        if errors:
            response["metadata"]["errors"] = errors
            # If there's an error but we still got an answer, note that
            if response["answer"]:
                response["metadata"]["partial_success"] = True
        
        return response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error during pipeline execution: {str(e)}")
        print(error_traceback)
        
        # Calculate elapsed time even for failures
        elapsed_time = time.time() - start_time
        
        # Return error information in a standardized format
        return {
            "answer": f"An error occurred: {str(e)}",
            "memory": memory,  # Return the original memory
            "retrieved_documents": [],
            "metadata": {
                "elapsed_time": elapsed_time,
                "repo_identifier": repo_identifier,
                "error": str(e),
                "error_traceback": error_traceback,
                "timestamp": datetime.now().isoformat(),
            },
            "error": True
        }

def run_retrieval_only_pipeline(state: RAGState, collection_name: str) -> RAGState:
    """
    A simplified pipeline that skips loading and indexing steps, 
    only running retrieval and generation for existing collections.
    """
    import traceback
    import chromadb
    
    # Make a copy of the state to avoid modifying the input
    result_state = RAGState.from_dict(state.to_dict())
    
    print(f"\n==== RETRIEVAL-ONLY PIPELINE START ====")
    print(f"Input state: {result_state}")
    
    # Ensure we have access to the collection
    try:
        # Create vectorstore to ensure proper access to ChromaDB
        persistent_dir = get_persistent_dir()
        client = chromadb.PersistentClient(path=persistent_dir)
        
        # Verify collection exists
        collections = client.list_collections()
        if not any(c.name == collection_name for c in collections):
            print(f"Collection '{collection_name}' not found, cannot proceed with retrieval-only mode.")
            print("Falling back to full indexing. Please wait...")
            return debug_rag_pipeline(state)
            
        # Create the vectorstore for the collection
        vectorstore = Chroma(
            client=client, 
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings()
        )
        
        # Set vectorstore and collection name in the state
        result_state['vectorstore'] = vectorstore
        result_state['collection_name'] = collection_name
        print(f"Successfully connected to collection '{collection_name}'")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        print("Falling back to full indexing. Please wait...")
        return debug_rag_pipeline(state)
    
    # Only run the necessary nodes
    node_sequence = [
        ("retrieve", retrieve_node),
        ("generate", generate_node),
        ("memory", memory_node)
    ]
    
    # Run each node in sequence with verbose debugging
    for node_name, node_func in node_sequence:
        try:
            print(f"\n==== Running node: {node_name} ====")
            
            # Execute the node
            start_time = time.time()
            
            # Make sure we catch any exceptions inside the node execution
            try:
                result_state = node_func(result_state)
                elapsed_time = time.time() - start_time
                
                # Log key stats after node execution
                print(f"Node {node_name} completed in {elapsed_time:.2f} seconds")
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"\n==== ERROR in node {node_name} after {elapsed_time:.2f} seconds ====")
                print(f"Error: {str(e)}")
                traceback.print_exc()
                
                # Add error information to the state
                result_state[f"error_{node_name}"] = str(e)
                result_state[f"error_traceback_{node_name}"] = traceback.format_exc()
                
                print(f"Non-critical node {node_name} failed. Attempting to continue pipeline.")
                continue
            
            # Report node-specific stats
            if node_name == "retrieve" and "relevant_documents" in result_state:
                print(f"Retrieved {len(result_state['relevant_documents'])} documents")
                for i, doc in enumerate(result_state['relevant_documents'][:3]):  # Show top 3
                    print(f"  Top result {i+1}: {doc.metadata.get('file_path', 'unknown')} ({len(doc.page_content)} chars)")
            elif node_name == "generate" and "answer" in result_state:
                answer_preview = result_state['answer'][:100] + "..." if len(result_state['answer']) > 100 else result_state['answer']
                print(f"Generated answer: {answer_preview}")
                
        except Exception as e:
            # This should not happen as we're already catching exceptions inside the inner try-except block
            print(f"\n==== UNEXPECTED ERROR in retrieval_only_pipeline for node {node_name} ====")
            print(f"Error: {str(e)}")
            traceback.print_exc()
    
    print(f"\n==== RETRIEVAL-ONLY PIPELINE COMPLETE ====")
    
    return result_state

# --- ChromaDB Helper Functions ---
def get_chroma_client(persist_dir: str = None, max_retries: int = 3):
    """
    Creates and returns a ChromaDB client with the specified persistence directory.
    
    Args:
        persist_dir: Directory for ChromaDB persistence. If None, uses the default from config.
        max_retries: Maximum number of retries if client creation fails.
    
    Returns:
        A ChromaDB PersistentClient instance
    """
    import chromadb
    import time
    import random
    import os
    import shutil
    
    if persist_dir is None:
        persist_dir = get_persistent_dir()
    
    # Ensure the directory exists
    os.makedirs(persist_dir, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Clean up any potential lock files that could cause issues
            lock_file = os.path.join(persist_dir, ".lock")
            if os.path.exists(lock_file):
                try:
                    # Check age of lock file - if it's more than 5 minutes old, assume it's stale
                    lock_age = time.time() - os.path.getmtime(lock_file)
                    if lock_age > 300:  # 5 minutes in seconds
                        os.remove(lock_file)
                        print(f"Removed stale lock file (age: {lock_age:.1f}s): {lock_file}")
                    else:
                        print(f"Lock file exists but appears active (age: {lock_age:.1f}s), waiting...")
                        time.sleep(random.uniform(0.5, 2.0))  # Random wait to avoid collisions
                except Exception as e:
                    print(f"Warning: Issue with lock file {lock_file}: {e}")
            
            # Create the client
            client = chromadb.PersistentClient(path=persist_dir)
            
            # Test the client by listing collections
            try:
                client.list_collections()
                # Success! Return the client
                return client
            except Exception as e:
                print(f"Client created but failed basic test: {e}")
                raise
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = random.uniform(0.5, 2.0) * (attempt + 1)  # Exponential backoff with randomization
                print(f"Error creating ChromaDB client (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts to create ChromaDB client failed")
                
                # Last resort: if this is the final attempt, try to recreate the ChromaDB directory
                try:
                    # Backup the existing directory
                    backup_dir = f"{persist_dir}_backup_{int(time())}"
                    if os.path.exists(persist_dir):
                        shutil.copytree(persist_dir, backup_dir)
                        print(f"Created backup of ChromaDB directory at {backup_dir}")
                        
                        # Remove the directory and recreate
                        temp_dir = f"{persist_dir}_old_{int(time())}"
                        os.rename(persist_dir, temp_dir)
                        os.makedirs(persist_dir, exist_ok=True)
                        
                        # Final attempt with fresh directory
                        client = chromadb.PersistentClient(path=persist_dir)
                        
                        # Test the client and return it if successful
                        client.list_collections()
                        print("Successfully created client with fresh ChromaDB directory")
                        return client
                except Exception as final_e:
                    print(f"Final attempt with directory reset failed: {final_e}")
                
                # If we get here, all attempts have failed
                raise ValueError(f"Failed to create ChromaDB client after multiple attempts: {e}")
    
    # This should never be reached due to the exception in the last iteration
    raise ValueError("Failed to create ChromaDB client: unknown error")

def get_or_create_chroma_collection(collection_name: str, client=None, recreate: bool = False):
    """
    Gets or creates a ChromaDB collection with the specified name.
    
    Args:
        collection_name: Name of the collection to get or create
        client: ChromaDB client instance. If None, creates a new one.
        recreate: If True, deletes and recreates the collection if it exists.
    
    Returns:
        A ChromaDB collection
    """
    if client is None:
        client = get_chroma_client()
    
    try:
        # Check if collection exists using a try/except to catch '_type' KeyError
        collections = []
        exists = False
        
        try:
            collections = client.list_collections()
            exists = any(c.name == collection_name for c in collections)
        except KeyError as e:
            # Handle '_type' KeyError which indicates a ChromaDB format issue
            if "'_type'" in str(e):
                print(f"ChromaDB format error detected: {e}")
                recreate = True
                exists = False
            else:
                raise e
        except Exception as e:
            print(f"Error listing collections: {e}")
            recreate = True
            exists = False
        
        if exists and recreate:
            print(f"Deleting existing collection '{collection_name}'")
            try:
                client.delete_collection(collection_name)
                import time
                time.sleep(0.5)  # Small delay to ensure deletion completes
                exists = False
            except Exception as e:
                print(f"Warning: Failed to delete collection: {e}")
                # Try recreating the client
                client = get_chroma_client()
                exists = False
        
        if exists and not recreate:
            print(f"Using existing collection '{collection_name}'")
            return client.get_collection(name=collection_name)
        else:
            print(f"Creating new collection '{collection_name}'")
            return client.create_collection(name=collection_name)
    except Exception as e:
        print(f"Error managing collection '{collection_name}': {e}")
        raise ValueError(f"Failed to manage ChromaDB collection: {e}")

# Explicitly add a time import for debug_rag_pipeline
import time

# Export main functions
__all__ = ["run_rag_pipeline", "rag_graph", "ConversationMemory", "RAGState"] 