#!/usr/bin/env python
"""
Test script to verify that wiki directories are created with the standardized naming convention.
This script creates empty directory structures to verify that repository IDs with special characters
are properly normalized to use underscores instead.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the current directory to the Python path so imports work
sys.path.append('.')

from api.langgraph.wiki_structure import normalize_repo_id, get_wiki_data_dir, get_repo_data_dir

def test_wiki_dir_creation(repo_id: str):
    """Test directory creation with proper normalized ID.
    
    Args:
        repo_id: Repository ID with special characters to test normalization
    """
    print(f"\n===== Testing wiki directory creation for '{repo_id}' =====")
    
    # Get the normalized version
    normalized_id = normalize_repo_id(repo_id)
    print(f"Original repo ID: {repo_id}")
    print(f"Normalized repo ID: {normalized_id}")
    
    # Check if special characters were properly replaced with underscores
    special_chars = ['.', '-', '/', '+', '&', '@', '#', '$', '%', '^', '*', '(', ')', '=', '!']
    special_chars_in_original = [c for c in special_chars if c in repo_id]
    special_chars_in_normalized = [c for c in special_chars if c in normalized_id]
    
    if special_chars_in_original and not special_chars_in_normalized:
        print("✅ Normalization properly replaced ALL special characters with underscores")
    else:
        if special_chars_in_normalized:
            print("❌ Normalization did not replace these special characters: " + 
                  ", ".join(special_chars_in_normalized))
        else:
            print("✅ Normalization properly replaced special characters with underscores")
    
    # Check if normalized ID contains only alphanumeric and underscore
    if normalized_id.replace('_', '').isalnum():
        print("✅ Normalized ID contains only alphanumeric characters and underscores")
    else:
        print("❌ Normalized ID contains characters other than alphanumeric and underscores")
        non_alnum_underscores = [c for c in normalized_id if not (c.isalnum() or c == '_')]
        print(f"   Problematic characters: {non_alnum_underscores}")
    
    # Get the directory that would be created
    wiki_dir = get_repo_data_dir(repo_id)
    wiki_base = os.path.basename(wiki_dir)
    print(f"Wiki directory path: {wiki_dir}")
    print(f"Wiki directory name: {wiki_base}")
    
    # Clean up: Remove the directories if they were created during the test
    # This also tests that get_repo_data_dir creates directories
    if os.path.exists(wiki_dir):
        try:
            # Remove and recreate empty directory
            shutil.rmtree(wiki_dir)
            print(f"Removed test directory: {wiki_dir}")
        except Exception as e:
            print(f"Error removing directory: {e}")
    
    # Test with get_wiki_data_dir to check the base paths
    wiki_data_dir = get_wiki_data_dir()
    expected_wiki_dir = os.path.join(wiki_data_dir, "wikis", normalized_id)
    print(f"Expected wiki directory: {expected_wiki_dir}")
    
    # Check if it matches our expectation
    if wiki_dir == expected_wiki_dir:
        print("✅ Directory path matches expected normalized path")
    else:
        print("❌ Directory path does not match expected normalized path")
        print(f"  Expected: {expected_wiki_dir}")
        print(f"  Actual: {wiki_dir}")

if __name__ == "__main__":
    # Test with repository ID containing dots and hyphens
    test_repo_id = "customs.exchange-rate-main"
    test_wiki_dir_creation(test_repo_id)
    
    # Try with a more complex ID to ensure all special characters are handled
    complex_repo_id = "test.repo-with.special/chars+symbols&more!"
    test_wiki_dir_creation(complex_repo_id) 