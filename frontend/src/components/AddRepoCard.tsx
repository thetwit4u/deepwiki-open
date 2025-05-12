import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface AddRepoCardProps {
  onRepoAdded: () => void;
}

export default function AddRepoCard({ onRepoAdded }: AddRepoCardProps) {
  const [repoUrl, setRepoUrl] = useState("");
  const [progress, setProgress] = useState<number>(0);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [polling, setPolling] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [wikiStructure, setWikiStructure] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<string>("gemini");

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (repoUrl && polling) {
      interval = setInterval(() => {
        fetch(`/api/progress?repo=${encodeURIComponent(repoUrl)}`)
          .then((res) => {
            if (!res.ok) {
              throw new Error(`Server returned ${res.status}: ${res.statusText}`);
            }
            return res.json();
          })
          .then((data) => {
            setProgress(data.progress || 0);
            setStatus(data.status || "");
            setLog(data.log || []);
            setWikiStructure(data.wikiStructure || null);
            
            // Check for error status and handle it
            if (data.status === "error") {
              setError(data.error || "Error during processing");
              setPolling(false);
            }
            
            if (data.status === "done") {
              setPolling(false);
              setTimeout(() => {
                setProgress(0);
                setStatus("");
                setRepoUrl("");
                setLog([]);
                setWikiStructure(null);
                onRepoAdded();
              }, 1200);
            }
          })
          .catch((err) => {
            console.error("Progress fetch error:", err);
            setError(`Failed to fetch progress: ${err.message}`);
            // Don't stop polling right away on network errors, might be temporary
            if ((err.message || "").includes("Failed to fetch") && polling) {
              // Continue polling despite network error
            } else {
              setPolling(false);
            }
          });
      }, 400); // Poll twice as fast
    }
    return () => clearInterval(interval);
  }, [repoUrl, polling, onRepoAdded]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setStatus("");
    setProgress(0);
    setPolling(false);
    if (!repoUrl.trim()) {
      setError("Please enter a repository URL or path.");
      return;
    }
    try {
      // Call new backend pipeline endpoint
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_URL || 'http://localhost:8001';
      
      // Build request body with model configuration
      const requestBody: any = { 
        repo_url: repoUrl, 
        model: selectedModel === "deterministic" ? "deterministic" : "gemini"
      };
      
      const res = await fetch(`${backendUrl}/start-wiki-generation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      
      if (!res.ok) {
        const err = await res.text();
        setError(`Failed to start wiki generation: ${err}`);
        return;
      }
      setPolling(true);
    } catch (err) {
      setError("Failed to start repository scan.");
    }
  };

  return (
    <Card className="p-6 mb-6 max-w-xl">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <label className="font-medium">Add New Repository</label>
        <input
          type="text"
          className="border rounded px-3 py-2 text-sm"
          placeholder="Repository URL or local path"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
          disabled={polling}
        />
        
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium">Wiki Structure Generation</label>
          <select
            className="border rounded px-3 py-2 text-sm"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={polling}
          >
            <option value="gemini">AI-Powered (Gemini)</option>
            <option value="deterministic">Fixed Structure (No AI)</option>
          </select>
          <div className="text-xs text-gray-500">
            {selectedModel === "deterministic" 
              ? "Uses fixed structure based on repository type, no AI generation."
              : "Uses AI to intelligently generate wiki structure based on repository content."}
          </div>
        </div>
        
        <Button type="submit" disabled={polling || !repoUrl.trim()}>
          {polling ? "Processing..." : "Add Repository"}
        </Button>
        
        {error && <div className="text-red-500 text-sm">{error}</div>}
        {polling && (
          <div className="mt-2">
            <div className="text-xs text-gray-600 mb-2 font-semibold">{status ? `Current step: ${status}` : "Processing..."}</div>
            <ol className="text-xs text-gray-700 mb-2 list-decimal list-inside">
              {log.map((step, i) => (
                <li
                  key={i}
                  className={
                    i === log.length - 1
                      ? "font-bold text-black"
                      : "opacity-70 text-gray-500"
                  }
                >
                  {step}
                </li>
              ))}
            </ol>
            {wikiStructure && (
              <div className="text-green-600 text-xs font-semibold mt-2">Wiki structure generated!</div>
            )}
          </div>
        )}
      </form>
    </Card>
  );
} 