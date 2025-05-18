#!/usr/bin/env python
"""
Verify DeepWiki's embedding consistency across chat and wiki generation components.

This script runs both test_chat.py and test_wiki_embeddings.py in sequence to verify
that both components use ollama_nomic embeddings consistently.
"""

import os
import sys
import subprocess
import time

def run_test(test_script, description):
    """Run a test script and capture its output and exit code."""
    print(f"\n{'=' * 80}")
    print(f"Running {description}...")
    print(f"{'=' * 80}")
    
    try:
        process = subprocess.run(
            [sys.executable, test_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Print the output
        print(process.stdout)
        if process.stderr:
            print("STDERR:", process.stderr)
        
        # Return True if the test passed (exit code 0), False otherwise
        return process.returncode == 0
    except Exception as e:
        print(f"Error running {test_script}: {e}")
        return False

def main():
    """Run all embedding verification tests."""
    print("\nDeepWiki Embedding Consistency Verification")
    print("===========================================")
    print("This script verifies that both the chat and wiki generation components")
    print("use the ollama_nomic embedding provider consistently.")
    
    # Use a sample repository ID that exists in your system
    repo_id = "customs.exchange-rate-main"
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    
    print(f"\nUsing repository ID: {repo_id}")
    
    # Run the embedding generation test first (if needed)
    print("\nStep 1: Ensure embeddings exist for the repository")
    if not os.path.exists("generate_embeddings.py"):
        print("generate_embeddings.py not found. Skipping embedding generation step.")
    else:
        try:
            print("Running generate_embeddings.py to ensure embeddings exist...")
            subprocess.run(
                [sys.executable, "generate_embeddings.py", repo_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            print("Embeddings generation completed.")
            # Give ChromaDB a moment to finish writing
            time.sleep(1)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
    
    # Run the tests
    print("\nStep 2: Verify chat uses ollama_nomic embeddings")
    chat_success = run_test("test_chat.py", "Chat embedding verification")
    
    print("\nStep 3: Verify wiki generation uses ollama_nomic embeddings")
    wiki_success = run_test("test_wiki_embeddings.py", "Wiki embedding verification")
    
    # Print overall result
    print("\n" + "=" * 80)
    print("Embedding Consistency Verification Results:")
    print(f"Chat component: {'✅ PASSED' if chat_success else '❌ FAILED'}")
    print(f"Wiki component: {'✅ PASSED' if wiki_success else '❌ FAILED'}")
    
    if chat_success and wiki_success:
        print("\n✅ SUCCESS: Both components consistently use ollama_nomic embeddings!")
        return 0
    else:
        print("\n❌ ERROR: Embedding inconsistency detected. See logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 