from api.langgraph.state import RAGState
from api.langgraph.filters import custom_file_filter, should_include_file, get_directory_exclusion_patterns, get_file_extension_inclusions, get_binary_extensions_exclusions
import os
from langchain_community.document_loaders import GitLoader, DirectoryLoader, TextLoader
from langchain_core.documents import Document

def load_documents_node(state: RAGState) -> RAGState:
    """
    Loads documents from a Git repository (URL) or a local directory path.
    Expects 'repo_identifier' in state (either a URL or a local path).
    Stores a list of langchain Document objects in state['documents'].
    """
    repo_identifier = state.get("repo_identifier")
    print(f"State keys in load_documents_node: {state.keys()}")
    print(f"Repository identifier: {repo_identifier}")
    if not repo_identifier:
        raise ValueError("No 'repo_identifier' provided in state.")
    if isinstance(repo_identifier, str):
        repo_identifier = repo_identifier.rstrip("/")
        if repo_identifier.startswith("~"):
            repo_identifier = os.path.expanduser(repo_identifier)
        if not (repo_identifier.startswith("http://") or repo_identifier.startswith("https://")):
            repo_identifier = os.path.abspath(repo_identifier)
    if repo_identifier.startswith("http://") or repo_identifier.startswith("https://"):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            try:
                import subprocess
                possible_branches = []
                try:
                    cmd = ["git", "ls-remote", "--symref", repo_identifier, "HEAD"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and "ref: refs/heads/" in result.stdout:
                        for line in result.stdout.splitlines():
                            if line.startswith("ref: refs/heads/"):
                                branch_line = line.split()
                                branch = branch_line[1].split("refs/heads/")[-1].strip()
                                if branch:
                                    possible_branches.append(branch)
                                    break
                except (subprocess.SubprocessError, IndexError, TimeoutError):
                    pass
                for fallback in ["main", "master"]:
                    if fallback not in possible_branches:
                        possible_branches.append(fallback)
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
        if not os.path.exists(repo_identifier) or not os.path.isdir(repo_identifier):
            raise ValueError(f"Local path does not exist or is not a directory: {repo_identifier}")
        print(f"Loading documents from local directory: {repo_identifier}")
        try:
            def is_valid_file(file_path):
                return should_include_file(file_path)
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
                for pattern in get_directory_exclusion_patterns():
                    if pattern in file_path.split(os.sep):
                        skipped_files['excluded_dirs'] += 1
                        return False
                _, ext = os.path.splitext(file_path.lower())
                if ext in get_binary_extensions_exclusions():
                    skipped_files['excluded_extensions'] += 1
                    return False
                if ext not in get_file_extension_inclusions():
                    skipped_files['excluded_extensions'] += 1
                    return False
                try:
                    if os.path.getsize(file_path) > 1024 * 1024:
                        skipped_files['too_large'] += 1
                        return False
                except (OSError, IOError):
                    skipped_files['other'] += 1
                    return False
                return True
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
        if 'total' in skipped_files:
            print(f"Skipped {skipped_files['total']} files in total")
        else:
            print(f"Skipped files summary:")
            print(f"  - Cache files (__pycache__, .pyc, etc.): {skipped_files['cache_files']}")
            print(f"  - Files in excluded directories: {skipped_files['excluded_dirs']}")
            print(f"  - Files with excluded extensions: {skipped_files['excluded_extensions']}")
            print(f"  - Files too large (>1MB): {skipped_files['too_large']}")
            print(f"  - Other skipped files: {skipped_files['other']}")
    for doc in documents:
        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
            doc.metadata = {}
        if 'file_path' not in doc.metadata:
            doc.metadata['file_path'] = doc.metadata.get('source', 'unknown')
        if 'type' not in doc.metadata:
            ext = os.path.splitext(doc.metadata['file_path'])[1].lower()
            doc.metadata['type'] = ext.lstrip('.')
        doc.metadata['repository_id'] = repo_identifier
    state["documents"] = documents
    print(f"Loaded {len(documents)} documents")
    return state

# Usage Example
if __name__ == "__main__":
    state = RAGState()
    state["repo_identifier"] = "/path/to/repo"
    result = load_documents_node(state)
    print(result.keys()) 