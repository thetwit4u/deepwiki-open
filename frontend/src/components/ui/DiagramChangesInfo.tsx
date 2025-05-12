import React, { useState } from 'react';

interface DiagramChangesInfoProps {
  changes: string[];
  className?: string;
}

export function DiagramChangesInfo({ changes, className = '' }: DiagramChangesInfoProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  if (!changes || changes.length === 0) {
    return null;
  }
  
  return (
    <div className={`text-xs border rounded-md overflow-hidden ${className}`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3 py-2 bg-blue-50 text-left flex items-center justify-between hover:bg-blue-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="font-medium text-blue-700">
            {changes.length} change{changes.length !== 1 ? 's' : ''} made to diagram
          </span>
        </div>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className={`h-4 w-4 text-blue-600 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {isExpanded && (
        <div className="px-3 py-2 bg-white border-t border-blue-100">
          <ul className="space-y-1 text-gray-700">
            {changes.map((change, index) => (
              <li key={index} className="flex items-start gap-1.5">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 text-blue-500 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <span>{change}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
} 