# Mermaid Diagram Auto-Fix Updates for Frontend

## Overview

We've enhanced the backend API to support up to 3 attempts to automatically fix Mermaid diagram syntax errors, and we've improved the contextual awareness of each fix attempt. This document outlines what frontend changes are needed to fully leverage these improvements.

## Required Frontend Updates

1. **Increase Maximum Fix Attempts**
   - Update the maximum number of fix attempts from 2 to 3
   - Display all three attempts in the UI when they occur

2. **Pass Error Messages to Subsequent Attempts**
   - When a fix attempt fails, include the error message from the failed attempt in the next request
   - This provides additional context to the LLM for better fix results

## Implementation Details

### API Changes

The `FixMermaidRequest` model now supports:
```typescript
interface FixMermaidRequest {
  diagram: string;            // The Mermaid diagram code with errors
  error: string;              // The error message from Mermaid.js
  attempt: number;            // Current attempt number (1, 2, or 3)
}
```

### Frontend Implementation Example

```typescript
// Example implementation
let maxAttempts = 3; // Updated from 2 to 3
let currentAttempt = 1;
let errors = []; // Store errors for context

async function fixMermaidDiagram(diagram, errorMessage) {
  // Store the error for future attempts
  errors.push(errorMessage);
  
  // Make API request including the error history
  const response = await fetch('/api/fix-mermaid', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      diagram: diagram,
      error: errorMessage,
      attempt: currentAttempt
    })
  });
  
  const result = await response.json();
  
  // Try to render the fixed diagram
  try {
    renderMermaidDiagram(result.fixed_diagram);
    // Success! Show the user what changes were made
    displayChanges(result.changes_made);
    return true;
  } catch (newError) {
    // If still failing and attempts remain, try again
    currentAttempt++;
    if (currentAttempt <= maxAttempts) {
      // Show the user that we're making another attempt
      displayAttemptProgress(currentAttempt, maxAttempts);
      return fixMermaidDiagram(result.fixed_diagram, newError.message);
    } else {
      // Give up and let the user edit manually
      displayMaxAttemptsReached();
      return false;
    }
  }
}

// Display progress to the user
function displayAttemptProgress(currentAttempt, maxAttempts) {
  const progressElement = document.getElementById('mermaid-fix-progress');
  progressElement.textContent = `Fix attempt ${currentAttempt} of ${maxAttempts}...`;
  progressElement.style.display = 'block';
}

// Show changes to the user
function displayChanges(changes) {
  const changesElement = document.getElementById('mermaid-changes');
  changesElement.innerHTML = `<h4>Changes Made (Attempt ${currentAttempt}):</h4>
    <ul>${changes.map(change => `<li>${change}</li>`).join('')}</ul>`;
  changesElement.style.display = 'block';
}
```

## UI Considerations

1. **Progress Indicator**
   - Show which attempt is currently being processed (e.g., "Fix attempt 2 of 3...")
   - Use a progress bar or similar visual to indicate multiple attempts are being made

2. **Changes List**
   - After each attempt, show the user what changes were made to the diagram
   - Ideally, highlight the exact portions of code that were modified

3. **Final Fallback**
   - If all three attempts fail, provide a user-friendly editor for manual fixes
   - Consider showing the user the specific error message to help them fix it manually

## Testing

Please test the following scenarios:
1. Diagrams with syntax errors that are fixed on the first attempt
2. Diagrams requiring multiple attempts to fix
3. Diagrams that cannot be fixed after all three attempts
4. The user experience when waiting for multiple fix attempts

Let us know if you encounter any issues or have questions! 