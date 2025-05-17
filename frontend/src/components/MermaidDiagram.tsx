import React, { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

let mermaidId = 0;
export default function MermaidDiagram({ code, onDiagramFixed }: { code: string, onDiagramFixed?: (originalCode: string, fixedCode: string) => void }) {
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
  const [errorHistory, setErrorHistory] = useState<string[]>([]);
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
      setErrorHistory(prev => [...prev, errorMessage]);
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
    setIsFixing(true);
    try {
      let errorContext = error;
      if (fixAttempts > 0 && errorHistory.length > 0) {
        errorContext = `Current error: ${error}\nPrevious errors: ${errorHistory.join("; ")}`;
      }
      const response = await fetch('/api/fix-mermaid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ diagram: currentCode, error: errorContext, attempt: fixAttempts + 1 }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
        throw new Error(errorData.error || 'Failed to get diagram fix');
      }
      const data = await response.json();
      if (data.fixed_diagram) {
        const oldCode = currentCode;
        const newCode = data.fixed_diagram;
        if (data.changes_made && data.changes_made.length > 0) setChangesMade(data.changes_made);
        setCurrentCode(data.fixed_diagram);
        setFixAttempts(prev => prev + 1);
        try {
          mermaid.initialize({ startOnLoad: false, theme: "neutral" });
          const testId = `test-${id}`;
          await mermaid.render(testId, newCode.trim());
          setIsFixed(true);
          if (onDiagramFixed) onDiagramFixed(oldCode, newCode);
        } catch (renderErr) {
          const newErrorMessage = renderErr instanceof Error ? renderErr.message : String(renderErr);
          setErrorHistory(prev => [...prev, newErrorMessage]);
          if (fixAttempts + 1 >= MAX_FIX_ATTEMPTS) setError(newErrorMessage);
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      if (err instanceof Error) {
        const newErrorMessage = `Failed to optimize diagram: ${err.message}`;
        setError(newErrorMessage);
        setErrorHistory(prev => [...prev, newErrorMessage]);
      }
    } finally {
      setIsFixing(false);
    }
  };

  useEffect(() => {
    if (error && !isFixing && fixAttempts < MAX_FIX_ATTEMPTS) {
      const timer = setTimeout(() => { handleAutoFix(); }, 500);
      return () => clearTimeout(timer);
    }
  }, [error, isFixing, fixAttempts]);

  if (!isClient) return <div className="my-4 text-gray-400">[Mermaid diagram will render on client]</div>;
  const toggleErrorDetails = () => { setExpandedError(!expandedError); };
  return (
    <div className="my-4">
      <div ref={ref} className="mb-2" />
      {error && (
        <div className={`mt-3 border rounded-md overflow-hidden ${isFixed ? 'border-green-200' : 'border-blue-200'}`}>
          <div className={`px-3 py-2 flex items-center justify-between ${isFixed ? 'bg-green-50' : 'bg-blue-50'}`}>
            <div className="flex items-center">
              {isFixing ? (
                <svg className="animate-spin h-4 w-4 text-blue-500 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
            {error && !isFixed && (
              <button 
                onClick={toggleErrorDetails}
                className="text-xs text-blue-600 hover:text-blue-800 hover:underline focus:outline-none"
              >
                {expandedError ? 'Hide details' : 'Show details'}
              </button>
            )}
          </div>
          {expandedError && (
            <div className="px-3 py-2 bg-gray-50 border-t border-blue-100">
              <p className="text-xs text-gray-700 font-medium mb-1">Current error:</p>
              <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto whitespace-pre-wrap text-gray-700">
                {error}
              </pre>
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
      {isFixed && changesMade.length > 0 && (
        <div className="mt-2">
          <ul className="list-disc list-inside text-xs text-green-700">
            {changesMade.map((change, idx) => (
              <li key={idx}>{change}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
} 