import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import mermaid from "mermaid";
import { Card } from "@/components/ui/card";
import { Badge } from "../components/ui/badge";
import { DiagramChangesInfo } from "./ui/DiagramChangesInfo";
import { DirectoryTree, isDirectoryStructure } from "./ui/DirectoryTree";

let mermaidId = 0;
function MermaidDiagram({ code, onDiagramFixed }: { code: string, onDiagramFixed?: (originalCode: string, fixedCode: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const id = React.useMemo(() => `mermaid-svg-${mermaidId++}`, []);
  const [isClient, setIsClient] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fixAttempts, setFixAttempts] = useState(0);
  const [isFixing, setIsFixing] = useState(false);
  const [originalCode, setOriginalCode] = useState(code);
  const [currentCode, setCurrentCode] = useState(code);
  const [isFixed, setIsFixed] = useState(false);
  const [expandedError, setExpandedError] = useState(false);
  const [changesMade, setChangesMade] = useState<string[]>([]);
  // Store error history for better context between attempts
  const [errorHistory, setErrorHistory] = useState<string[]>([]);
  // Maximum number of allowed fix attempts
  const MAX_FIX_ATTEMPTS = 3;

  useEffect(() => {
    setIsClient(typeof window !== "undefined");
    setOriginalCode(code);
    setCurrentCode(code);
    setError(null);
    setFixAttempts(0);
    setIsFixed(false);
    setExpandedError(false);
    setChangesMade([]);
    setErrorHistory([]);
  }, [code]);

  const renderDiagram = async (diagramCode: string) => {
    if (!isClient || !ref.current) return;
    
    try {
      setError(null);
      mermaid.initialize({ startOnLoad: false, theme: "neutral" });
      const { svg } = await mermaid.render(id, diagramCode.trim() || "graph TD;A-->B;");
      if (ref.current) ref.current.innerHTML = svg;
    } catch (err) {
      console.error("Mermaid rendering error:", err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(errorMessage);
      
      // Add error to history for future context
      setErrorHistory(prev => [...prev, errorMessage]);
      
      // Show a more informative placeholder when diagram fails to render
      if (ref.current) {
        ref.current.innerHTML = `
          <div class="flex flex-col items-center justify-center p-4 rounded bg-gray-50 border border-gray-200">
            <div class="flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <h3 class="ml-2 text-base font-medium text-gray-700">Optimizing Diagram</h3>
            </div>
            <p class="mt-2 text-sm text-gray-600">Automatic diagram repair in progress...</p>
          </div>
        `;
      }
    }
  };

  useEffect(() => {
    renderDiagram(currentCode);
  }, [currentCode, id, isClient]);

  const handleAutoFix = async () => {
    if (fixAttempts >= MAX_FIX_ATTEMPTS || !error) return;
    
    console.log("[FIX DEBUG] Starting handleAutoFix, current error:", error);
    console.log("[FIX DEBUG] Error history:", errorHistory);
    setIsFixing(true);
    try {
      // Create a comprehensive error context that includes the current error
      // and, for subsequent attempts, the history of previous errors
      let errorContext = error;
      if (fixAttempts > 0 && errorHistory.length > 0) {
        // Format a more detailed error message with history
        errorContext = `Current error: ${error}\nPrevious errors: ${errorHistory.join("; ")}`;
        console.log("[FIX DEBUG] Using expanded error context with history");
      }
      
      const response = await fetch('/api/fix-mermaid', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          diagram: currentCode,
          error: errorContext, // Send the enhanced error context
          attempt: fixAttempts + 1
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
        console.error('[FIX DEBUG] Error response:', errorData);
        throw new Error(errorData.error || 'Failed to get diagram fix');
      }
      
      const data = await response.json();
      console.log("[FIX DEBUG] Received fix data:", data);
      
      if (data.fixed_diagram) {
        // Save the old and new code before updating state
        const oldCode = currentCode;
        const newCode = data.fixed_diagram;
        console.log("[FIX DEBUG] Got fixed diagram, old length:", oldCode.length, "new length:", newCode.length);
        
        // If there were changes reported in the response, log them
        if (data.changes_made && data.changes_made.length > 0) {
          console.log("[FIX DEBUG] Changes made:", data.changes_made);
          setChangesMade(data.changes_made);
        }
        
        // Update the state
        setCurrentCode(data.fixed_diagram);
        setFixAttempts(prev => prev + 1);
        
        // Try rendering the new diagram to verify it works
        try {
          console.log("[FIX DEBUG] Testing if fixed diagram renders correctly");
          mermaid.initialize({ startOnLoad: false, theme: "neutral" });
          const testId = `test-${id}`;
          await mermaid.render(testId, newCode.trim());
          
          // If we get here, the diagram rendered successfully without errors
          console.log("[FIX DEBUG] Fix successful! Diagram renders without errors. Calling onDiagramFixed");
          setIsFixed(true);
          if (onDiagramFixed) {
            onDiagramFixed(oldCode, newCode);
          } else {
            console.log("[FIX DEBUG] WARNING: onDiagramFixed callback is not defined!");
          }
        } catch (renderErr) {
          console.log("[FIX DEBUG] New diagram still has errors, not saving:", renderErr);
          // The new diagram still has errors, so we don't save it
          
          // Add the new error to our history
          const newErrorMessage = renderErr instanceof Error ? renderErr.message : String(renderErr);
          setErrorHistory(prev => [...prev, newErrorMessage]);
          
          // If this was the final attempt, show a more detailed error
          if (fixAttempts + 1 >= MAX_FIX_ATTEMPTS) {
            setError(newErrorMessage);
          }
        }
      } else {
        console.error('[FIX DEBUG] Response missing fixed_diagram field:', data);
        throw new Error('Invalid response format');
      }
    } catch (err) {
      console.error('[FIX DEBUG] Error fixing diagram:', err);
      
      // Update the error state with the caught error message
      if (err instanceof Error) {
        const newErrorMessage = `Failed to optimize diagram: ${err.message}`;
        setError(newErrorMessage);
        setErrorHistory(prev => [...prev, newErrorMessage]);
      }
    } finally {
      setIsFixing(false);
    }
  };

  // Trigger auto-fix automatically when there's an error
  useEffect(() => {
    if (error && !isFixing && fixAttempts < MAX_FIX_ATTEMPTS) {
      // Add a small delay to avoid immediate retries
      const timer = setTimeout(() => {
        handleAutoFix();
      }, 500);
      
      return () => clearTimeout(timer);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [error, isFixing, fixAttempts]);

  if (!isClient) return <div className="my-4 text-gray-400">[Mermaid diagram will render on client]</div>;
  
  // Create a function to toggle the expanded error state
  const toggleErrorDetails = () => {
    setExpandedError(!expandedError);
  };
  
  return (
    <div className="my-4">
      <div ref={ref} className="mb-2" />
      
      {/* Status interface for diagram repair */}
      {error && (
        <div className={`mt-3 border rounded-md overflow-hidden ${isFixed ? 'border-green-200' : 'border-blue-200'}`}>
          {/* Header bar */}
          <div className={`px-3 py-2 flex items-center justify-between ${isFixed ? 'bg-green-50' : 'bg-blue-50'}`}>
            <div className="flex items-center">
              {isFixing ? (
                <svg className="animate-spin h-4 w-4 text-blue-500 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : isFixed ? (
                <svg className="h-4 w-4 text-green-500 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="h-4 w-4 text-blue-500 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
              <span className={`text-sm font-medium ${isFixed ? 'text-green-700' : 'text-blue-700'}`}>
                {isFixing ? 'Optimizing diagram...' : 
                 isFixed ? 'Diagram fixed successfully' : 
                 fixAttempts >= MAX_FIX_ATTEMPTS ? 'Diagram partially optimized' : 'Diagram needs optimization'}
              </span>
            </div>
            
            {/* Show error details button - only when there's still an error */}
            {error && !isFixed && (
              <button 
                onClick={toggleErrorDetails}
                className="text-xs text-blue-600 hover:text-blue-800 hover:underline focus:outline-none"
              >
                {expandedError ? 'Hide details' : 'Show details'}
              </button>
            )}
          </div>
          
          {/* Error details panel - collapsed by default */}
          {expandedError && (
            <div className="px-3 py-2 bg-gray-50 border-t border-blue-100">
              <p className="text-xs text-gray-700 font-medium mb-1">Current error:</p>
              <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto whitespace-pre-wrap text-gray-700">
                {error}
              </pre>
              
              {/* Show error history if we have made at least one attempt */}
              {errorHistory.length > 0 && fixAttempts > 0 && (
                <div className="mt-3">
                  <p className="text-xs text-gray-700 font-medium mb-1">Previous errors:</p>
                  <div className="text-xs bg-gray-100 p-2 rounded overflow-x-auto text-gray-700 max-h-24">
                    {errorHistory.map((err, index) => (
                      <div key={index} className="mb-1 pb-1 border-b border-gray-200 last:border-0">
                        <span className="text-blue-600 font-medium">Attempt {index + 1}: </span>
                        {err}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <p className="text-xs text-gray-500 mt-2">
                {fixAttempts < MAX_FIX_ATTEMPTS ? 
                  `Automatic repair in progress (${fixAttempts}/${MAX_FIX_ATTEMPTS} attempts made)...` : 
                  `Maximum repair attempts reached (${MAX_FIX_ATTEMPTS}/${MAX_FIX_ATTEMPTS}). The diagram may need manual adjustment.`}
              </p>
              
              {/* Add helpful tips for common errors */}
              <div className="mt-3 text-xs border-t border-gray-200 pt-2">
                <p className="font-medium text-gray-700">Common fixes:</p>
                <ul className="list-disc list-inside text-gray-600 mt-1">
                  <li>Add quotes around node labels with spaces: <code className="bg-gray-200 px-1">A["Label with spaces"]</code></li>
                  <li>Check graph direction (should be TD, LR, etc.)</li>
                  <li>Verify all subgraphs have end statements</li>
                  <li>Remove any non-diagram text or comments</li>
                </ul>
              </div>
            </div>
          )}
          
          {/* Status footer */}
          <div className={`px-3 py-1.5 text-xs ${isFixed ? 'bg-green-50 text-green-600' : 'bg-blue-50 text-blue-600'}`}>
            {isFixing ? (
              <span>Attempt {fixAttempts + 1}/{MAX_FIX_ATTEMPTS}...</span>
            ) : fixAttempts >= MAX_FIX_ATTEMPTS && !isFixed ? (
              <span>Optimization attempted ({MAX_FIX_ATTEMPTS}/{MAX_FIX_ATTEMPTS}). Some issues may remain.</span>
            ) : isFixed ? (
              <span>Diagram successfully repaired in {fixAttempts} {fixAttempts === 1 ? 'attempt' : 'attempts'}.</span>
            ) : (
              <span>Preparing to optimize diagram...</span>
            )}
          </div>
        </div>
      )}
      
      {/* Show changes made when diagram is fixed and there are changes to report */}
      {isFixed && changesMade.length > 0 && (
        <DiagramChangesInfo 
          changes={changesMade} 
          className="mt-2"
        />
      )}
    </div>
  );
}

interface SectionContentProps {
  content: string;
  metadata?: {
    title?: string;
    description?: string;
    tags?: string[];
    section_id?: string;
    generated_at?: string;
    repo_url?: string;
    [key: string]: any;
  };
  onContentChanged?: (newContent: string) => void;
}

export default function SectionContent({ content, metadata = {}, onContentChanged }: SectionContentProps) {
  // The content should already be clean from the backend, but we'll do a final check
  const [processedContent, setProcessedContent] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{text: string, type: 'success' | 'error'} | null>(null);
  const [hasPendingChanges, setHasPendingChanges] = useState(false);
  const [localMetadata, setLocalMetadata] = useState(metadata);
  
  // Add effect to extract repo_url from query parameters
  useEffect(() => {
    if (typeof window !== "undefined") {
      const extractRepoFromQuery = () => {
        const urlParams = new URLSearchParams(window.location.search);
        const repoFromQuery = urlParams.get('repo') || urlParams.get('repo_url') || urlParams.get('repository');
        
        console.log("[URL DEBUG] Query parameters:", {
          repo: urlParams.get('repo'),
          repo_url: urlParams.get('repo_url'),
          repository: urlParams.get('repository'),
          section: urlParams.get('section'),
          section_id: urlParams.get('section_id')
        });
        
        if (repoFromQuery) {
          console.log("[URL DEBUG] Found repo_url in query params:", repoFromQuery);
          setLocalMetadata(prev => ({
            ...prev,
            repo_url: repoFromQuery
          }));
        }
        
        // Also look for section_id in query if needed
        const sectionFromQuery = urlParams.get('section') || urlParams.get('section_id');
        if (sectionFromQuery && !localMetadata.section_id) {
          console.log("[URL DEBUG] Found section_id in query params:", sectionFromQuery);
          setLocalMetadata(prev => ({
            ...prev,
            section_id: sectionFromQuery
          }));
        }
      };
      
      extractRepoFromQuery();
      
      // Also try to extract repo from URL path
      const pathParts = window.location.pathname.split('/');
      console.log("[URL DEBUG] Path parts:", pathParts);
      
      // Look for pattern: /wiki/{repo}/...
      const wikiIndex = pathParts.indexOf('wiki');
      if (wikiIndex >= 0 && wikiIndex < pathParts.length - 1) {
        const possibleRepo = pathParts[wikiIndex + 1];
        if (possibleRepo && !possibleRepo.includes('section')) {
          console.log("[URL DEBUG] Found possible repo_url in path:", possibleRepo);
          setLocalMetadata(prev => ({
            ...prev,
            repo_url: prev.repo_url || possibleRepo
          }));
        }
      }
    }
  }, []);
  
  // Key effect: Reset state when content or metadata changes (page switch)
  useEffect(() => {
    console.log("[PAGE SWITCH] Content or metadata changed, resetting component state");
    console.log("[PAGE SWITCH] New metadata:", metadata);
    
    // Reset all stateful values to match the new page
    setProcessedContent("");  // Will be reprocessed in the content effect
    setIsSaving(false);
    setSaveMessage(null);
    setHasPendingChanges(false);
    setLocalMetadata(metadata);  // Use the new metadata directly
    
    // Log the new state for debugging
    console.log("[PAGE SWITCH] State reset completed");
  }, [content, metadata]);  // This effect runs when switching pages
  
  // Function to extract repo_url from the document's context
  const extractRepoFromContext = () => {
    if (typeof window === "undefined") return;
    
    // Look for breadcrumb navigation
    const breadcrumbs = document.querySelectorAll('.breadcrumb a');
    console.log("[CONTEXT DEBUG] Found breadcrumbs:", breadcrumbs.length);
    
    for (let i = 0; i < breadcrumbs.length; i++) {
      const element = breadcrumbs[i];
      const href = element.getAttribute('href');
      const text = element.textContent;
      console.log(`[CONTEXT DEBUG] Breadcrumb ${i}:`, { href, text });
      
      // Look for repo-like text in breadcrumb
      if (text && (text.includes('/') || text.includes('.'))) {
        console.log("[CONTEXT DEBUG] Found potential repo in breadcrumb:", text);
        return text;
      }
      
      // Look for repo in href
      if (href && href.includes('/wiki/')) {
        const hrefParts = href.split('/wiki/');
        if (hrefParts.length > 1 && hrefParts[1]) {
          const repoFromHref = hrefParts[1].split('/')[0];
          console.log("[CONTEXT DEBUG] Found potential repo in breadcrumb href:", repoFromHref);
          return repoFromHref;
        }
      }
    }
    
    // Try looking at window title
    if (document.title) {
      const titleParts = document.title.split(' - ');
      if (titleParts.length > 1) {
        const potentialRepo = titleParts[titleParts.length - 1];
        console.log("[CONTEXT DEBUG] Found potential repo in document title:", potentialRepo);
        return potentialRepo;
      }
    }
    
    return null;
  };
  
  // Add context extraction to our main metadata effect
  useEffect(() => {
    if (!localMetadata.repo_url) {
      const repoFromContext = extractRepoFromContext();
      if (repoFromContext) {
        console.log("[CONTEXT DEBUG] Setting repo_url from context:", repoFromContext);
        setLocalMetadata(prev => ({
          ...prev,
          repo_url: repoFromContext
        }));
      }
    }
  }, [localMetadata.repo_url]);
  
  // Update localMetadata when metadata prop changes
  useEffect(() => {
    console.log("[METADATA UPDATE] Metadata prop changed:", metadata);
    setLocalMetadata(metadata);
  }, [metadata]);
  
  // Function to save updated content back to the server
  const saveUpdatedContent = async (updatedContent: string) => {
    console.log("[SAVE DEBUG] saveUpdatedContent called!");
    console.log("[SAVE DEBUG] Local Metadata:", JSON.stringify(localMetadata));
    console.log("[SAVE DEBUG] Prop Metadata:", JSON.stringify(metadata));
    
    // Always use the most recent metadata
    const metadataToUse = metadata || localMetadata;
    
    if (!metadataToUse.section_id || !metadataToUse.repo_url) {
      console.error("[SAVE DEBUG] Cannot save content: missing section_id or repo_url in metadata", metadataToUse);
      setSaveMessage({
        text: "Cannot save content: missing required metadata (section_id or repo_url)",
        type: 'error'
      });
      return;
    }
    
    console.log(`[SAVE DEBUG] Saving updated content for section ${metadataToUse.section_id} of repo ${metadataToUse.repo_url}`);
    console.log(`[SAVE DEBUG] Content length: ${updatedContent.length} characters`);
    
    setIsSaving(true);
    setSaveMessage(null);
    
    try {
      // Use the standard Next.js API route
      const apiUrl = "/api/update-section-content";
      
      console.log(`[SAVE DEBUG] Making API request to ${apiUrl}`);
      const requestBody = {
        repo_url: metadataToUse.repo_url,
        section_id: metadataToUse.section_id,
        content: updatedContent,
        metadata: metadataToUse
      };
      console.log("[SAVE DEBUG] Request body:", JSON.stringify(requestBody).substring(0, 200) + "...");
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      console.log(`[SAVE DEBUG] API response status: ${response.status}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[SAVE DEBUG] Error response: ${response.status} - ${errorText}`);
        throw new Error(`Failed to save content: ${response.statusText} (${response.status})`);
      }
      
      const responseData = await response.json();
      console.log(`[SAVE DEBUG] API response data:`, responseData);
      
      setSaveMessage({
        text: "Content updated successfully with fixed diagram",
        type: 'success'
      });
      
      // Update local state to reflect changes are saved
      setHasPendingChanges(false);
      
      // Notify parent if provided
      if (onContentChanged) {
        onContentChanged(updatedContent);
      }
    } catch (error) {
      console.error("[SAVE DEBUG] Error saving content:", error);
      setSaveMessage({
        text: `Error saving content: ${error instanceof Error ? error.message : String(error)}`,
        type: 'error'
      });
    } finally {
      setIsSaving(false);
      
      // Clear message after 5 seconds
      setTimeout(() => {
        setSaveMessage(null);
      }, 5000);
    }
  };
  
  // Handle fixed Mermaid diagrams
  const handleDiagramFixed = (originalCode: string, fixedCode: string) => {
    console.log("[DIAGRAM DEBUG] handleDiagramFixed called with:", {
      originalLength: originalCode.length,
      fixedLength: fixedCode.length,
      originalStart: originalCode.substring(0, 30),
      fixedStart: fixedCode.substring(0, 30)
    });
    
    // Log the current metadata for debugging
    console.log("[DIAGRAM DEBUG] Current metadata when fixing diagram:", metadata);
    console.log("[DIAGRAM DEBUG] Current localMetadata when fixing diagram:", localMetadata);
    console.log("[DIAGRAM DEBUG] URL path:", window.location.pathname);
    console.log("[DIAGRAM DEBUG] URL query:", window.location.search);
    
    // Normalize diagram codes for better matching
    const normalizedOriginal = originalCode.trim();
    const normalizedFixed = fixedCode.trim();
    
    // Special handling for specific error pattern
    let finalFixedCode = normalizedFixed;
    
    // Check for common syntax issue where labels with spaces need quotes
    if (normalizedFixed.includes("A[Clone Repository]") && !normalizedFixed.includes("A[\"Clone Repository\"]")) {
      console.log("[DIAGRAM DEBUG] Applying additional fix for unquoted node labels");
      // Replace all node labels with spaces with properly quoted ones
      finalFixedCode = normalizedFixed
        .replace(/([A-Z])\[([^\]]+\s+[^\]]+)\]/g, '$1["$2"]')
        .replace(/I\[Commit Changes\]/g, 'I["Commit Changes"]');
      
      console.log("[DIAGRAM DEBUG] After additional quoting fix:", finalFixedCode);
    }
    
    if (normalizedOriginal === finalFixedCode) {
      console.log("[DIAGRAM DEBUG] Diagram code unchanged, not saving");
      return;
    }
    
    // Replace the broken diagram in the content with the fixed one
    // We'll try different strategies to find the exact diagram
    
    let updatedContent = processedContent;
    let diagramReplaced = false;
    
    // Strategy 1: Look for exact match with code fences
    const exactMermaidPattern = new RegExp(
      '```mermaid\\s*\\n' + // Opening code fence
      escapeRegExp(normalizedOriginal) + // Original diagram code
      '\\s*\\n```', // Closing code fence
      'g'
    );
    
    // Debug log for the error message
    if (normalizedOriginal.includes("...git Changes") || finalFixedCode.includes("...git Changes")) {
      console.log("[DIAGRAM DEBUG] Found problematic diagram with 'git Changes' text");
      console.log("[DIAGRAM DEBUG] Original diagram:\n", normalizedOriginal);
      console.log("[DIAGRAM DEBUG] Fixed diagram:\n", finalFixedCode);
    }
    
    if (updatedContent.match(exactMermaidPattern)) {
      // If we find an exact match for the diagram, replace it
      console.log("[DIAGRAM DEBUG] Found exact diagram match, replacing");
      updatedContent = updatedContent.replace(
        exactMermaidPattern,
        `\`\`\`mermaid\n${finalFixedCode}\n\`\`\``
      );
      diagramReplaced = true;
    } 
    // Strategy 2: Look for just the diagram content without code fences
    else {
      const contentOnlyPattern = new RegExp(
        escapeRegExp(normalizedOriginal),
        'g'
      );
      
      if (updatedContent.match(contentOnlyPattern)) {
        console.log("[DIAGRAM DEBUG] Found content-only match, replacing");
        updatedContent = updatedContent.replace(
          contentOnlyPattern,
          finalFixedCode
        );
        diagramReplaced = true;
      }
      // Strategy 3: Find any mermaid diagram
      else {
        console.log("[DIAGRAM DEBUG] No exact match, searching for any mermaid diagram");
        const anyMermaidPattern = /```mermaid\n[\s\S]*?\n```/g;
        const mermaidMatches = updatedContent.match(anyMermaidPattern) || [];
        
        if (mermaidMatches.length > 0) {
          // Replace the first diagram we find
          console.log("[DIAGRAM DEBUG] Found", mermaidMatches.length, "mermaid diagrams, replacing first one");
          const firstMatch = mermaidMatches[0];
          if (firstMatch) {
            updatedContent = updatedContent.replace(
              firstMatch,
              `\`\`\`mermaid\n${finalFixedCode}\n\`\`\``
            );
            diagramReplaced = true;
          }
        } else {
          // If we still can't find a diagram, append the fixed one
          console.log("[DIAGRAM DEBUG] No mermaid diagrams found, appending fixed diagram");
          updatedContent = updatedContent + 
            "\n\n## Fixed Diagram\n\n" +
            `\`\`\`mermaid\n${finalFixedCode}\n\`\`\`\n`;
          diagramReplaced = true;
        }
      }
    }
    
    // Strategy 4: Look for standalone "mermaid" text that might be an incorrectly formatted diagram
    if (!diagramReplaced) {
      const standaloneMermaidPattern = /\n(mermaid)\s*\n/g;
      if (updatedContent.match(standaloneMermaidPattern)) {
        console.log("[DIAGRAM DEBUG] Found standalone mermaid text, replacing with properly formatted diagram");
        updatedContent = updatedContent.replace(
          standaloneMermaidPattern,
          `\n\`\`\`mermaid\n${finalFixedCode}\n\`\`\`\n`
        );
        diagramReplaced = true;
      }
    }
    
    // Update the displayed content
    console.log("[DIAGRAM DEBUG] Setting processed content with updated diagram");
    setProcessedContent(updatedContent);
    
    // Get the proper combined metadata, ensuring we prioritize the current props
    // Important: We need to combine the metadata properly
    const currentMetadata = {
      ...(localMetadata || {}),  // Start with local metadata as base
      ...(metadata || {})        // Override with prop metadata (which takes priority)
    };
    
    // Re-extract repo from URL if not in metadata
    if (!currentMetadata.repo_url) {
      const urlParams = new URLSearchParams(window.location.search);
      const repoFromQuery = urlParams.get('repo') || urlParams.get('repo_url') || urlParams.get('repository');
      
      if (repoFromQuery) {
        console.log("[DIAGRAM DEBUG] Found repo_url in query params during save:", repoFromQuery);
        currentMetadata.repo_url = repoFromQuery;
      } else {
        // Try path
        const pathParts = window.location.pathname.split('/');
        const wikiIndex = pathParts.indexOf('wiki');
        if (wikiIndex >= 0 && wikiIndex < pathParts.length - 1) {
          const possibleRepo = pathParts[wikiIndex + 1];
          if (possibleRepo && !possibleRepo.includes('section')) {
            console.log("[DIAGRAM DEBUG] Found repo_url in path during save:", possibleRepo);
            currentMetadata.repo_url = possibleRepo;
          }
        }
      }
    }
    
    console.log("[DIAGRAM DEBUG] Combined metadata for saving:", currentMetadata);
    console.log("[DIAGRAM DEBUG] Checking metadata fields specifically:", {
      "section_id exists": !!currentMetadata.section_id,
      "repo_url exists": !!currentMetadata.repo_url,
      "section_id value": currentMetadata.section_id,
      "repo_url value": currentMetadata.repo_url
    });
    
    // Check if we have the required fields
    const canSaveAutomatically = !!(
      currentMetadata && 
      currentMetadata.section_id && 
      currentMetadata.repo_url
    );
    
    console.log(`[DIAGRAM DEBUG] Can save automatically: ${canSaveAutomatically}`, {
      section_id: currentMetadata.section_id,
      repo_url: currentMetadata.repo_url
    });
    
    // Always try to save immediately if we have the metadata, otherwise show the save button
    if (canSaveAutomatically) {
      console.log("[DIAGRAM DEBUG] Metadata present, saving content to server");
      
      // Call the save function with the combined metadata
      const saveRequestBody = {
        repo_url: currentMetadata.repo_url,
        section_id: currentMetadata.section_id,
        content: updatedContent,
        metadata: currentMetadata
      };
      
      console.log("[DIAGRAM DEBUG] Save request body:", saveRequestBody);
      
      // Use the standard API route
      fetch('/api/update-section-content', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(saveRequestBody),
      })
      .then(response => {
        console.log(`[DIAGRAM DEBUG] Save response status: ${response.status}`);
        if (response.ok) {
          setSaveMessage({
            text: "Content updated successfully with fixed diagram",
            type: 'success'
          });
          setHasPendingChanges(false);
        } else {
          return response.text().then(text => {
            throw new Error(`Failed to save: ${response.status} ${response.statusText} - ${text}`);
          });
        }
      })
      .catch(error => {
        console.error("[DIAGRAM DEBUG] Error saving content:", error);
        setSaveMessage({
          text: `Error saving content: ${error.message}`,
          type: 'error'
        });
        setHasPendingChanges(true);
      });
    } else {
      console.log("[DIAGRAM DEBUG] Metadata missing, showing local success message only");
      // If we can't save to server, at least show a success message to the user and set pending changes
      setHasPendingChanges(true);
      setSaveMessage({
        text: "Diagram fixed (local only). Click 'Save Changes' to save to server.",
        type: 'success'
      });
      
      // Clear message after 5 seconds
      setTimeout(() => {
        setSaveMessage(null);
      }, 5000);
    }
    
    // Also notify parent component if callback provided
    if (onContentChanged) {
      onContentChanged(updatedContent);
    }
  };
  
  // Helper function to escape special characters in a string for use in regex
  const escapeRegExp = (string: string) => {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  };
  
  // Helper function to extract the graph type from mermaid code
  const getGraphType = (code: string): string | null => {
    // Look for common graph types
    const graphMatch = code.match(/^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph|journey)\b/m);
    return graphMatch ? graphMatch[1] : null;
  };

  useEffect(() => {
    // Function to ensure content is clean
    const ensureCleanContent = (text: string): string => {
      if (!text) return "";
      
      let cleaned = text;
      
      // Enhanced frontmatter removal - check for various patterns
      
      // Standard YAML frontmatter with triple dashes
      if (cleaned.startsWith('---')) {
        const endIndex = cleaned.indexOf('---', 3);
        if (endIndex > 0) {
          cleaned = cleaned.substring(endIndex + 3).trim();
        }
      }
      
      // Handle frontmatter without proper triple dash delimiters
      // This pattern looks for title:, description:, tags:, etc. at the beginning of content
      const frontmatterRegex = /^\s*(title|description|tags|section_id|generated_at|repo_url):\s+.*(\n\s*[a-z_]+:.+)*\n/i;
      if (frontmatterRegex.test(cleaned)) {
        // If we detect frontmatter-like pattern at start, try to skip to the first proper markdown heading or text
        const lines = cleaned.split("\n");
        let contentStartIndex = 0;
        let inBulletList = false;
        
        // Skip initial frontmatter-like lines
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();
          
          // Check if we're entering a bullet list that might be part of frontmatter
          if (line.startsWith("tags:") || (line === "" && i > 0 && lines[i-1].trim().endsWith("tags:"))) {
            inBulletList = true;
            contentStartIndex = i + 1;
            continue;
          }
          
          // Handle bullet points that are part of frontmatter
          if (inBulletList && (line.startsWith("•") || line.startsWith("-") || line.startsWith("*"))) {
            contentStartIndex = i + 1;
            continue;
          }
          
          // If we find a non-bullet line after being in a bullet list, we're likely out of the frontmatter
          if (inBulletList && line !== "" && !line.startsWith("•") && !line.startsWith("-") && !line.startsWith("*")) {
            inBulletList = false;
            // Only break if this line is actually content (not more frontmatter)
            if (!line.match(/^[a-z_]+:\s/i)) {
              break;
            }
          }
          
          // Continue skipping until we find a line that doesn't look like frontmatter
          if (line === "" || /^[a-z_]+:\s+.*$/i.test(line)) {
            contentStartIndex = i + 1;
            continue;
          }
          
          // Special case for the section_id line with generation timestamp
          if (line.includes("section_id:") && line.includes("generated_at:")) {
            contentStartIndex = i + 1;
            continue;
          }
          
          // Special case for the placeholder generation time line
          if (line.includes("Placeholder, replace with actual generation time") || 
              line.match(/#\s*Placeholder/i)) {
            contentStartIndex = i + 1;
            continue;
          }
          
          // Stop when we find a heading or non-frontmatter-like content
          if (line.startsWith("#") || (!line.includes(":") && !inBulletList)) {
            break;
          }
        }
        
        if (contentStartIndex > 0 && contentStartIndex < lines.length) {
          cleaned = lines.slice(contentStartIndex).join("\n").trim();
        }
      }
      
      // Final clean-up for any remaining markdown code blocks or yaml blocks
      // This is just a safety check in case the backend missed something
      if (cleaned.startsWith('```yaml')) {
        const endIndex = cleaned.indexOf('\n```');
        if (endIndex > 0) {
          cleaned = cleaned.substring(endIndex + 4).trim();
        }
      } else if (cleaned.startsWith('```markdown')) {
        const endIndex = cleaned.indexOf('\n```');
        if (endIndex > 0) {
          cleaned = cleaned.substring(endIndex + 4).trim();
        }
      }
      
      // Check for JSON wrapped content (a common issue with nested quotes)
      if (cleaned.startsWith('{"content":')) {
        try {
          const parsed = JSON.parse(cleaned);
          if (parsed.content) {
            cleaned = parsed.content;
          }
        } catch (e) {
          // Invalid JSON, ignore
        }
      }
      
      // Fix incorrectly formatted mermaid diagrams (without code fence)
      // Look for "mermaid\ngraph" pattern which is common in incorrectly formatted diagrams
      cleaned = cleaned.replace(/\n(mermaid)\s*\n(graph\s+[A-Z][A-Z])/g, "\n```mermaid\n$2");
      cleaned = cleaned.replace(/\n(mermaid)\s*\n(sequenceDiagram)/g, "\n```mermaid\n$2");
      cleaned = cleaned.replace(/\n(mermaid)\s*\n(flowchart\s+[A-Z][A-Z])/g, "\n```mermaid\n$2");
      
      // Add closing code fence for mermaid diagrams if it doesn't exist
      const mermaidBlocks = cleaned.match(/```mermaid\n[\s\S]*?(?:```|$)/g) || [];
      mermaidBlocks.forEach(block => {
        if (!block.endsWith("```")) {
          const fixedBlock = block + "\n```";
          cleaned = cleaned.replace(block, fixedBlock);
        }
      });
      
      return cleaned;
    };
    
    setProcessedContent(ensureCleanContent(content));
  }, [content]);
  
  // Update this when content is changed
  useEffect(() => {
    if (processedContent !== content && processedContent !== "") {
      setHasPendingChanges(true);
    } else {
      setHasPendingChanges(false);
    }
  }, [processedContent, content]);
  
  // Add verbose debugging for the component initialization
  useEffect(() => {
    console.log("[INIT DEBUG] SectionContent initialized with metadata:", metadata);
    console.log("[INIT DEBUG] Current window location:", {
      href: window?.location?.href,
      pathname: window?.location?.pathname,
      search: window?.location?.search,
      hash: window?.location?.hash
    });
    console.log("[INIT DEBUG] Initial content length:", content?.length);
  }, []);
  
  // Add an effect to try to extract metadata from the content if needed
  useEffect(() => {
    console.log("[METADATA DEBUG] Starting metadata extraction");
    console.log("[METADATA DEBUG] Current localMetadata:", localMetadata);
    console.log("[METADATA DEBUG] Current prop metadata:", metadata);
    
    if (!localMetadata.section_id || !localMetadata.repo_url) {
      // Try to extract from URL
      try {
        console.log("[METADATA DEBUG] Current URL path:", window.location.pathname);
        console.log("[METADATA DEBUG] Current URL:", window.location.href);
        const pathParts = window.location.pathname.split('/');
        console.log("[METADATA DEBUG] Path parts:", pathParts);
        
        // Manually check each path part for potential section_id or repo_url
        console.log("[METADATA DEBUG] Checking each path part individually:");
        pathParts.forEach((part, index) => {
          console.log(`[METADATA DEBUG] Path part ${index}: '${part}'`);
          
          // Look for pattern of repo name in path part
          if (part.includes('.') || part.includes('-') || part.includes('_')) {
            console.log(`[METADATA DEBUG] Potential repo_url candidate at index ${index}: '${part}'`);
          }
          
          // Look for potential section IDs (typically have hyphens)
          if (part.includes('-') && part.length > 8) {
            console.log(`[METADATA DEBUG] Potential section_id candidate at index ${index}: '${part}'`);
          }
        });
        
        // Expected format: /wiki/:repo_url/section/:section_id
        const sectionIndex = pathParts.indexOf('section');
        console.log("[METADATA DEBUG] Section index in path:", sectionIndex);
        
        if (sectionIndex > 0 && sectionIndex < pathParts.length - 1) {
          const section_id = pathParts[sectionIndex + 1];
          if (section_id) {
            console.log("[METADATA DEBUG] Extracted section_id from URL:", section_id);
            
            const repoIndex = pathParts.indexOf('wiki');
            console.log("[METADATA DEBUG] Wiki index in path:", repoIndex);
            
            if (repoIndex > 0 && repoIndex < pathParts.length - 1) {
              const repo_url = pathParts[repoIndex + 1];
              if (repo_url) {
                console.log("[METADATA DEBUG] Extracted repo_url from URL:", repo_url);
                
                setLocalMetadata(prevState => ({
                  ...prevState,
                  section_id: section_id,
                  repo_url: repo_url
                }));
                
                console.log("[METADATA DEBUG] Updated local metadata with URL data");
              }
            }
          }
        }
        
        // Fallback 1: Try to get parameters from query string
        const urlParams = new URLSearchParams(window.location.search);
        const querySection = urlParams.get('section_id');
        const queryRepo = urlParams.get('repo_url');
        
        console.log("[METADATA DEBUG] Query params:", {
          section_id: querySection,
          repo_url: queryRepo
        });
        
        if (querySection && queryRepo) {
          console.log("[METADATA DEBUG] Found metadata in query params");
          setLocalMetadata(prevState => ({
            ...prevState,
            section_id: querySection,
            repo_url: queryRepo
          }));
        }
        
        // Fallback 2: Try to find section ID in a more direct way from the last part of URL path
        if (!localMetadata.section_id) {
          const lastPathPart = pathParts[pathParts.length - 1];
          if (lastPathPart && lastPathPart !== 'section') {
            console.log("[METADATA DEBUG] Fallback: Using last URL path part as section_id:", lastPathPart);
            
            // Try to find repo part from pathname (usually after /wiki/)
            let repo_part = '';
            for (let i = 0; i < pathParts.length; i++) {
              if (pathParts[i] === 'wiki' && i + 1 < pathParts.length) {
                repo_part = pathParts[i + 1];
                break;
              }
            }
            
            if (repo_part) {
              console.log("[METADATA DEBUG] Fallback: Using path part after '/wiki/' as repo_url:", repo_part);
              
              setLocalMetadata(prevState => ({
                ...prevState,
                section_id: lastPathPart,
                repo_url: repo_part
              }));
              
              console.log("[METADATA DEBUG] Updated local metadata with fallback URL data");
            }
          }
        }
        
        // Fallback 3: Try to extract metadata from document title
        const docTitle = document.title;
        console.log("[METADATA DEBUG] Document title:", docTitle);
        if (docTitle && docTitle.includes(' - ')) {
          const titleParts = docTitle.split(' - ');
          if (titleParts.length >= 2) {
            const potentialSectionId = titleParts[0].trim().toLowerCase().replace(/\s+/g, '-');
            const potentialRepoUrl = titleParts[titleParts.length - 1].trim();
            
            console.log("[METADATA DEBUG] Potential metadata from title:", {
              section_id: potentialSectionId,
              repo_url: potentialRepoUrl
            });
            
            if (!localMetadata.section_id && !localMetadata.repo_url) {
              setLocalMetadata(prevState => ({
                ...prevState,
                section_id: potentialSectionId,
                repo_url: potentialRepoUrl
              }));
              console.log("[METADATA DEBUG] Updated metadata from document title");
            }
          }
        }
      } catch (error) {
        console.error("[METADATA DEBUG] Error extracting metadata from URL:", error);
      }
    } else {
      console.log("[METADATA DEBUG] Using existing metadata:", {
        section_id: localMetadata.section_id,
        repo_url: localMetadata.repo_url
      });
    }
  }, [localMetadata, metadata]);
  
  // Forcibly populate metadata if we have direct elements in the DOM we can use
  useEffect(() => {
    try {
      // Look for breadcrumb elements that might contain repo or section info
      const breadcrumbs = document.querySelectorAll('.breadcrumb a');
      console.log("[DOM DEBUG] Found breadcrumbs:", breadcrumbs.length);
      
      // Extract text from breadcrumbs for debugging
      Array.from(breadcrumbs).forEach((el, i) => {
        console.log(`[DOM DEBUG] Breadcrumb ${i}:`, el.textContent, el.getAttribute('href'));
      });
      
      // Look for section headers that might contain the section name
      const sectionHeadings = document.querySelectorAll('h1, h2');
      console.log("[DOM DEBUG] Found section headings:", sectionHeadings.length);
      
      // Extract text from headings for debugging
      Array.from(sectionHeadings).forEach((el, i) => {
        console.log(`[DOM DEBUG] Heading ${i}:`, el.textContent);
      });
    } catch (error) {
      console.error("[DOM DEBUG] Error inspecting DOM:", error);
    }
  }, []);
  
  return (
    <Card className="p-6 pt-4">
      {localMetadata.tags && localMetadata.tags.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {localMetadata.tags.map((tag, index) => (
            <Badge key={index} variant="outline">{tag}</Badge>
          ))}
        </div>
      )}
      
      {saveMessage && (
        <div className={`mb-2 p-2 rounded text-sm ${
          saveMessage.type === 'success' 
            ? 'bg-green-50 border border-green-200 text-green-700' 
            : 'bg-red-50 border border-red-200 text-red-700'
        }`}>
          {saveMessage.text}
        </div>
      )}
      
      <div className="-mt-1 prose prose-sm max-w-none dark:prose-invert prose-headings:font-bold prose-headings:text-gray-900 dark:prose-headings:text-gray-100 prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-p:first-of-type:mt-0 prose-ul:text-gray-700 dark:prose-ul:text-gray-300 prose-ol:text-gray-700 dark:prose-ol:text-gray-300 prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-code:text-pink-600 dark:prose-code:text-pink-400 prose-blockquote:text-gray-700 dark:prose-blockquote:text-gray-300 prose-a:text-blue-600 hover:prose-a:text-blue-800 dark:prose-a:text-blue-400 dark:hover:prose-a:text-blue-300">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw]}
          components={{
            h1: ({node, children, ...props}) => (
              <h1 className="text-2xl font-bold mt-1 mb-3" {...props}>{children}</h1>
            ),
            h2: ({node, children, ...props}) => (
              <h2 className="text-xl font-bold mt-3 mb-2" {...props}>{children}</h2>
            ),
            h3: ({node, children, ...props}) => (
              <h3 className="text-lg font-semibold mt-3 mb-2" {...props}>{children}</h3>
            ),
            h4: ({node, children, ...props}) => (
              <h4 className="text-base font-semibold mt-2 mb-1" {...props}>{children}</h4>
            ),
            p: ({node, className, children, ...props}) => {
              // Check if the paragraph content looks like a directory structure
              const paragraphContent = React.Children.toArray(children)
                .map(child => typeof child === 'string' ? child : '')
                .join('');
              
              if (isDirectoryStructure(paragraphContent)) {
                console.log("[PARAGRAPH DEBUG] Detected directory structure in paragraph");
                return <DirectoryTree content={paragraphContent} />;
              }
              
              // First paragraph is included with "prose-p:first-of-type" in the parent className
              return (
                <p className="my-2 leading-relaxed" {...props}>
                  {children}
                </p>
              );
            },
            ul: ({node, children, ...props}) => (
              <ul className="list-disc list-inside my-3 pl-2" {...props}>{children}</ul>
            ),
            ol: ({node, children, ...props}) => (
              <ol className="list-decimal list-inside my-3 pl-2" {...props}>{children}</ol>
            ),
            li: ({node, children, ...props}) => (
              <li className="my-1" {...props}>{children}</li>
            ),
            a: ({node, href, children, ...props}) => (
              <a 
                href={href} 
                className="text-blue-600 hover:text-blue-800 hover:underline" 
                target="_blank" 
                rel="noopener noreferrer" 
                {...props}
              >
                {children}
              </a>
            ),
            blockquote: ({node, children, ...props}) => (
              <blockquote 
                className="border-l-4 border-gray-300 pl-4 italic my-4 text-gray-700" 
                {...props}
              >
                {children}
              </blockquote>
            ),
            table: ({node, children, ...props}) => (
              <div className="overflow-x-auto my-4">
                <table className="min-w-full border-collapse border border-gray-300" {...props}>
                  {children}
                </table>
              </div>
            ),
            thead: ({node, children, ...props}) => (
              <thead className="bg-gray-100" {...props}>{children}</thead>
            ),
            tbody: ({node, children, ...props}) => (
              <tbody className="divide-y divide-gray-200" {...props}>{children}</tbody>
            ),
            tr: ({node, children, ...props}) => (
              <tr className="hover:bg-gray-50" {...props}>{children}</tr>
            ),
            th: ({node, children, ...props}) => (
              <th className="px-3 py-2 text-left font-medium text-gray-700 border border-gray-300" {...props}>
                {children}
              </th>
            ),
            td: ({node, children, ...props}) => (
              <td className="px-3 py-2 border border-gray-300" {...props}>{children}</td>
            ),
            code({node, className, children, ...props}) {
              const match = /language-(\w+)/.exec(className || "");
              
              // Handle Mermaid diagrams
              if (match && match[1] === "mermaid") {
                // Make sure to console.log to check if this is being called properly
                console.log("[CODE DEBUG] Rendering mermaid diagram, code length:", String(children).length);
                return <MermaidDiagram 
                  code={String(children).trim()} 
                  onDiagramFixed={(originalCode, fixedCode) => {
                    console.log("[CODE DEBUG] onDiagramFixed callback called!");
                    handleDiagramFixed(originalCode, fixedCode);
                  }}
                />;
              }
              
              // Handle directory structure listings (with or without language specification)
              const codeContent = String(children);
              if (isDirectoryStructure(codeContent)) {
                console.log("[CODE DEBUG] Detected directory structure listing");
                return <DirectoryTree 
                  content={codeContent} 
                  title={match ? `Directory Structure (${match[1]})` : "Directory Structure"}
                />;
              }
              
              if (match) {
                // For code blocks with language
                return (
                  <div className="my-4 rounded-md overflow-hidden">
                    <div className="bg-gray-800 text-gray-200 px-3 py-1 text-xs">
                      {match[1]}
                    </div>
                    <pre className="bg-gray-900 p-4 overflow-x-auto text-gray-200 text-sm">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  </div>
                );
              }
              
              // For inline code
              return (
                <code 
                  className="bg-gray-100 px-1.5 py-0.5 rounded text-pink-600 font-mono text-sm" 
                  {...props}
                >
                  {children}
                </code>
              );
            },
            strong: ({node, children, ...props}) => (
              <strong className="font-bold" {...props}>{children}</strong>
            ),
            em: ({node, children, ...props}) => (
              <em className="italic" {...props}>{children}</em>
            ),
            hr: ({node, ...props}) => (
              <hr className="my-6 border-t border-gray-300" {...props} />
            ),
            img: ({node, src, alt, ...props}) => (
              <img
                src={src}
                alt={alt || ""}
                className="max-w-full h-auto my-4 rounded-md"
                {...props}
              />
            ),
            pre: ({node, children, ...props}) => {
              // Check if pre content contains directory structure
              const preContent = React.Children.toArray(children)
                .map(child => {
                  if (React.isValidElement(child) && child.props) {
                    const childProps = child.props as { children?: React.ReactNode };
                    return typeof childProps.children === 'string' 
                      ? childProps.children 
                      : '';
                  }
                  return typeof child === 'string' ? child : '';
                })
                .join('');
              
              if (isDirectoryStructure(preContent)) {
                console.log("[PRE DEBUG] Detected directory structure in pre block");
                return <DirectoryTree content={preContent} />;
              }
              
              // Standard pre handling
              return (
                <pre className="bg-gray-900 p-4 overflow-x-auto text-gray-200 text-sm rounded-md my-4" {...props}>
                  {children}
                </pre>
              );
            },
          }}
        >
          {processedContent}
        </ReactMarkdown>
      </div>
      
      {localMetadata.generated_at && (
        <div className="mt-6 text-xs text-gray-400 text-right">
          Generated: {new Date(localMetadata.generated_at).toLocaleString()}
        </div>
      )}
    </Card>
  );
} 