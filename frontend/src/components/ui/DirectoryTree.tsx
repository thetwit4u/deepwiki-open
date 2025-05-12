import React from 'react';

interface DirectoryTreeProps {
  content: string;
  className?: string;
  title?: string;
}

export function DirectoryTree({ content, className = '', title = 'Directory Structure' }: DirectoryTreeProps) {
  // Pre-process the content to enhance directory structure visualization
  const processedContent = content
    // Ensure proper spacing after directory markers
    .replace(/(├──|└──)(?!\s)/g, '$1 ')
    // Ensure proper indentation
    .replace(/^(\s*)(├|└|│)/gm, '$1$2');
  
  return (
    <div className={`my-4 rounded-md overflow-hidden ${className}`}>
      <div className="bg-gray-800 text-gray-200 px-3 py-1 text-xs flex justify-between items-center">
        <span>{title}</span>
        <span className="text-gray-400 text-xs">Fixed-width font</span>
      </div>
      <pre className="bg-gray-900 p-4 overflow-x-auto text-gray-200 text-sm font-mono leading-relaxed whitespace-pre">
        {processedContent}
      </pre>
    </div>
  );
}

// Utility function to detect if a string looks like a directory structure
export function isDirectoryStructure(content: string): boolean {
  if (!content || typeof content !== 'string') return false;
  
  // Basic pattern matching for directory structures
  const dirPatterns = [
    // Tree-like symbols or starting with dot
    /^\s*[├│└─\.\s]+.*\/(?:\n|$)/,
    // Starts with dot and has tree-like lines
    /^\s*\.\n(?:\s*├── |\s*└── |\s*│\s+|\.\/)/m,
    // Multiple lines with tree-like symbols
    /^(?:\s*├── |\s*└── |\s*│\s+)/m
  ];
  
  // Additional checks to reduce false positives
  const hasTreeSymbols = content.includes('├') || content.includes('└') || content.includes('│');
  const hasDirectories = content.includes('/');
  const hasMultipleLines = content.includes('\n');
  const hasDotStart = content.trim().startsWith('.');
  
  // Calculate confidence score
  let confidence = 0;
  
  // Check basic patterns
  for (const pattern of dirPatterns) {
    if (pattern.test(content)) {
      confidence += 2;
    }
  }
  
  // Check additional signals
  if (hasTreeSymbols) confidence += 3;
  if (hasDirectories) confidence += 2;
  if (hasMultipleLines) confidence += 1;
  if (hasDotStart) confidence += 1;
  
  // Higher confidence threshold for plain text to avoid false positives
  return confidence >= 4;
} 