import os
import re
import hashlib
import chromadb
import time
import random
import shutil

def generate_collection_name(repo_identifier: str) -> str:
    """Generates a safe collection name from a repository identifier."""
    if repo_identifier.startswith("http://") or repo_identifier.startswith("https://"):
        path_parts = repo_identifier.rstrip("/").split("/")
        if len(path_parts) >= 2:
            owner = path_parts[-2]
            repo = path_parts[-1].replace(".git", "")
            prefix = f"git_{owner}_{repo}"
        else:
            repo_name = path_parts[-1].replace(".git", "")
            prefix = f"git_{repo_name}"
        prefix = re.sub(r'[^\w\-_]', '_', prefix)
        if len(prefix) > 60:
            short_hash = hashlib.md5(repo_identifier.encode()).hexdigest()[:10]
            prefix = f"{prefix[:50]}_{short_hash}"
    else:
        abs_path = os.path.abspath(repo_identifier)
        dir_name = os.path.basename(abs_path)
        dir_name = re.sub(r'[^\w\-_]', '_', dir_name)
        path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:10]
        prefix = f"local_{dir_name}_{path_hash}"
    return prefix

def get_persistent_dir() -> str:
    """Returns the directory for persistent ChromaDB storage"""
    home_dir = os.path.expanduser("~")
    chroma_dir = os.path.join(home_dir, ".deepwiki", "chromadb")
    os.makedirs(chroma_dir, exist_ok=True)
    return chroma_dir

def get_chroma_client(persist_dir: str = None, max_retries: int = 3):
    """Creates and returns a ChromaDB client with the specified persistence directory."""
    if persist_dir is None:
        persist_dir = get_persistent_dir()
    os.makedirs(persist_dir, exist_ok=True)
    for attempt in range(max_retries):
        try:
            lock_file = os.path.join(persist_dir, ".lock")
            if os.path.exists(lock_file):
                try:
                    lock_age = time.time() - os.path.getmtime(lock_file)
                    if lock_age > 300:
                        os.remove(lock_file)
                        print(f"Removed stale lock file (age: {lock_age:.1f}s): {lock_file}")
                    else:
                        print(f"Lock file exists but appears active (age: {lock_age:.1f}s), waiting...")
                        time.sleep(random.uniform(0.5, 2.0))
                except Exception as e:
                    print(f"Warning: Issue with lock file {lock_file}: {e}")
            client = chromadb.PersistentClient(path=persist_dir)
            try:
                client.list_collections()
                return client
            except Exception as e:
                print(f"Client created but failed basic test: {e}")
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = random.uniform(0.5, 2.0) * (attempt + 1)
                print(f"Error creating ChromaDB client (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts to create ChromaDB client failed")
                try:
                    backup_dir = f"{persist_dir}_backup_{int(time.time())}"
                    if os.path.exists(persist_dir):
                        shutil.copytree(persist_dir, backup_dir)
                        print(f"Created backup of ChromaDB directory at {backup_dir}")
                        temp_dir = f"{persist_dir}_old_{int(time.time())}"
                        os.rename(persist_dir, temp_dir)
                        os.makedirs(persist_dir, exist_ok=True)
                        client = chromadb.PersistentClient(path=persist_dir)
                        client.list_collections()
                        print("Successfully created client with fresh ChromaDB directory")
                        return client
                except Exception as final_e:
                    print(f"Final attempt with directory reset failed: {final_e}")
                raise ValueError(f"Failed to create ChromaDB client after multiple attempts: {e}")
    raise ValueError("Failed to create ChromaDB client: unknown error")

def get_or_create_chroma_collection(collection_name: str, client=None, recreate: bool = False):
    """Gets or creates a ChromaDB collection with the specified name."""
    if client is None:
        client = get_chroma_client()
    try:
        collections = []
        exists = False
        try:
            collections = client.list_collections()
            exists = any(c.name == collection_name for c in collections)
        except KeyError as e:
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
                time.sleep(0.5)
                exists = False
            except Exception as e:
                print(f"Warning: Failed to delete collection: {e}")
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

# Usage Example
if __name__ == "__main__":
    coll_name = generate_collection_name("https://github.com/test/repo.git")
    print(f"Generated Collection Name: {coll_name}")
    client = get_chroma_client()
    collection = get_or_create_chroma_collection(coll_name, client, recreate=True)
    print(f"Got/Created Collection: {collection.name}") 