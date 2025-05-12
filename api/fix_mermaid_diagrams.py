#!/usr/bin/env python3
"""
Script to fix incorrectly formatted Mermaid diagrams in wiki markdown files.
Scans all wiki content directories and applies fixes to ensure diagrams render properly.
"""

import os
import re
import sys
import argparse
import glob
from datetime import datetime

def process_file(file_path, dry_run=False):
    """Process a markdown file to fix Mermaid diagrams."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix Mermaid graph diagrams
    mermaid_pattern = re.compile(r'\n(mermaid)\s*\n(graph\s+[A-Z][A-Z][\s\S]*?)(?=\n##|\n#|\Z)', re.MULTILINE)
    if mermaid_pattern.search(content):
        content = mermaid_pattern.sub(r'\n```mermaid\n\2\n```\n', content)
        print(f"  Fixed graph diagram in {file_path}")
    
    # Fix Mermaid sequence diagrams
    sequence_pattern = re.compile(r'\n(mermaid)\s*\n(sequenceDiagram[\s\S]*?)(?=\n##|\n#|\Z)', re.MULTILINE)
    if sequence_pattern.search(content):
        content = sequence_pattern.sub(r'\n```mermaid\n\2\n```\n', content)
        print(f"  Fixed sequence diagram in {file_path}")
    
    # Fix Mermaid flowchart diagrams
    flowchart_pattern = re.compile(r'\n(mermaid)\s*\n(flowchart\s+[A-Z][A-Z][\s\S]*?)(?=\n##|\n#|\Z)', re.MULTILINE)
    if flowchart_pattern.search(content):
        content = flowchart_pattern.sub(r'\n```mermaid\n\2\n```\n', content)
        print(f"  Fixed flowchart diagram in {file_path}")
    
    # Check for incomplete mermaid diagrams (no ending ```)
    mermaid_blocks = re.findall(r'```mermaid\n[\s\S]*?(?=```|\n##|\n#|\Z)', content)
    for block in mermaid_blocks:
        if not block.endswith('```'):
            fixed_block = block + '\n```'
            content = content.replace(block, fixed_block)
            print(f"  Added missing closing fence to mermaid diagram in {file_path}")
    
    # Fix node labels - ensure text with spaces or special chars is properly quoted
    # Find all mermaid code blocks
    mermaid_blocks = re.findall(r'```mermaid\n([\s\S]*?)```', content)
    for i, block in enumerate(mermaid_blocks):
        fixed_block = block
        
        # Fix node labels in square brackets [Label] -> ["Label"] if they contain spaces or special chars
        # This pattern finds node definitions with unquoted labels containing spaces
        node_pattern = re.compile(r'\[([^\]"]*\s+[^\]"]*)\]')
        
        # Find all matches in this block
        matches = node_pattern.findall(fixed_block)
        if matches:
            # Replace each unquoted label with a quoted one
            for match in matches:
                if '"' not in match:  # Avoid double-quoting
                    original = f"[{match}]"
                    replacement = f'["{match}"]'
                    fixed_block = fixed_block.replace(original, replacement)
                    print(f"  Fixed unquoted node label in {file_path}: {original} -> {replacement}")
        
        # Remove non-mermaid text (explanatory comments, plain text) inside the mermaid block
        # Valid mermaid lines typically start with whitespace followed by identifiers, arrows, styling commands, etc.
        lines = fixed_block.split('\n')
        clean_lines = []
        has_non_mermaid_text = False
        
        for line in lines:
            # Skip empty lines or lines with just whitespace
            if not line.strip():
                continue
            
            # Keep lines that look like valid mermaid syntax (very simplified heuristic)
            if (
                # Basic mermaid commands or syntax elements
                re.match(r'\s*(?:graph|flowchart|sequenceDiagram|classDef|class|subgraph|end|style|linkStyle|click|%%)', line.strip()) or
                # Node definitions or connections
                re.match(r'\s*[A-Za-z0-9_-]+(\s*\[.*\]|\s*\(.*\)|\s*>.*|\s*--.*|\s*==.*|\s*-.*)$', line.strip()) or
                # Arrow or connection syntax
                re.match(r'\s*(?:-->|<-->|---|===|-.->|--x|--o|-\.->', line.strip()) or
                # Participant definitions for sequence diagrams
                re.match(r'\s*(?:participant|actor|note|loop|alt|opt|par|activate|deactivate)', line.strip())
            ):
                clean_lines.append(line)
            else:
                has_non_mermaid_text = True
                print(f"  Removed non-mermaid text from diagram in {file_path}: '{line.strip()}'")
        
        if has_non_mermaid_text:
            fixed_block = '\n'.join(clean_lines)
            content = content.replace(f"```mermaid\n{block}```", f"```mermaid\n{fixed_block}```")
            print(f"  Cleaned non-mermaid text from diagram in {file_path}")
            
        # Replace the original block with the fixed one for other changes
        elif fixed_block != block:
            content = content.replace(f"```mermaid\n{block}```", f"```mermaid\n{fixed_block}```")
    
    # Write back to file if changes were made and not in dry run mode
    if content != original_content and not dry_run:
        # Create backup
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Write fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Updated {file_path} (backup saved as {backup_path})")
    elif content != original_content:
        print(f"  Would update {file_path} (dry run)")
    
    return content != original_content

def main():
    parser = argparse.ArgumentParser(description='Fix incorrectly formatted Mermaid diagrams in wiki markdown files')
    parser.add_argument('--wiki-dir', type=str, default='wiki-data/wikis', help='Base directory containing wiki content')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run without modifying files')
    args = parser.parse_args()
    
    if not os.path.exists(args.wiki_dir):
        print(f"Error: Wiki directory '{args.wiki_dir}' does not exist")
        sys.exit(1)
    
    # Find all wiki repositories
    wiki_repos = [d for d in os.listdir(args.wiki_dir) if os.path.isdir(os.path.join(args.wiki_dir, d))]
    if not wiki_repos:
        print(f"No wiki repositories found in {args.wiki_dir}")
        sys.exit(0)
    
    total_files = 0
    fixed_files = 0
    
    for repo in wiki_repos:
        pages_dir = os.path.join(args.wiki_dir, repo, 'pages')
        if not os.path.exists(pages_dir):
            print(f"No pages directory found for {repo}")
            continue
        
        print(f"Processing wiki repository: {repo}")
        md_files = glob.glob(os.path.join(pages_dir, '*.md'))
        
        for md_file in md_files:
            total_files += 1
            if process_file(md_file, args.dry_run):
                fixed_files += 1
    
    print(f"\nSummary: Processed {total_files} files, fixed {fixed_files} files with Mermaid formatting issues")
    if args.dry_run:
        print("This was a dry run. No files were actually modified.")

if __name__ == "__main__":
    main() 