import os
import chromadb

# Get the path to the ChromaDB directory
persistent_dir = os.path.join(os.path.expanduser("~"), ".deepwiki", "chromadb")
print(f'ChromaDB directory: {persistent_dir}')
print(f'Directory exists: {os.path.exists(persistent_dir)}')

if os.path.exists(persistent_dir):
    # List directory contents
    print(f'Contents: {os.listdir(persistent_dir)}')
    
    # Create a client and list collections
    client = chromadb.PersistentClient(path=persistent_dir)
    collections = client.list_collections()
    print(f'Total collections: {len(collections)}')
    
    # Print all collection names
    print('All collection names:')
    for c in collections:
        print(f'- {c.name}') 