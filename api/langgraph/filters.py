import os
from typing import List, Set

def get_directory_exclusion_patterns() -> List[str]:
    """Returns patterns for directories to exclude"""
    return [
        'node_modules', 'bower_components', 'vendor', 'build', 'dist', 'target', 'out', '__pycache__', '.git', '.github', '.svn', '.idea', '.vscode', '.gradle', '.cache',
        'venv', 'virtualenv', '.env', 'env', '.venv', 'docs/build', 'site', 'Pods', 'logs', 'tmp', 'temp',
    ]

def get_file_extension_inclusions() -> Set[str]:
    """Returns file extensions to include"""
    return {
        '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.cpp', '.c', '.h', '.cs', '.rb', '.php', '.swift', '.kt', '.rs', '.scala',
        '.json', '.yaml', '.yml', '.toml', '.ini', '.xml', '.env.example',
        '.html', '.css', '.scss', '.sass', '.less',
        '.md', '.markdown', '.rst', '.txt',
        '.sh', '.bash', '.zsh', '.bat', '.ps1',
    }

def get_binary_extensions_exclusions() -> Set[str]:
    """Returns extensions of binary files to exclude"""
    return {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.svg', '.webp',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.webm', '.ogg', '.flac',
        '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz',
        '.exe', '.dll', '.so', '.dylib', '.jar', '.war', '.ear', '.whl', '.pyc', '.class',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.db', '.sqlite', '.sqlite3', '.mdb', '.csv',
        '.apk', '.ipa', '.ttf', '.otf', '.woff', '.woff2', '.eot',
    }

def should_include_file(file_path: str) -> bool:
    """Determines if a file should be included in document processing"""
    if '__pycache__' in file_path:
        return False
    for pattern in get_directory_exclusion_patterns():
        if pattern in file_path.split(os.sep):
            return False
    _, ext = os.path.splitext(file_path.lower())
    if ext in get_binary_extensions_exclusions():
        return False
    if ext in {'.pyc', '.pyo'}:
        return False
    if ext not in get_file_extension_inclusions():
        return False
    try:
        if os.path.getsize(file_path) > 1024 * 1024:
            return False
    except (OSError, IOError):
        return False
    return True

def custom_file_filter(file_path: str) -> bool:
    """Filter function for GitLoader (uses forward slashes)"""
    if '__pycache__' in file_path:
        return False
    for pattern in get_directory_exclusion_patterns():
        if pattern in file_path.split('/'):
            return False
    _, ext = os.path.splitext(file_path.lower())
    if ext in get_binary_extensions_exclusions():
        return False
    if ext in {'.pyc', '.pyo'}:
        return False
    if ext not in get_file_extension_inclusions():
        return False
    return True

# Usage Example
if __name__ == "__main__":
    print(should_include_file("src/main.py"))
    print(custom_file_filter("src/utils/helper.py")) 