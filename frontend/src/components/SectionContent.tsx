import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import mermaid from "mermaid";
import { Card } from "@/components/ui/card";
import { Badge } from "../components/ui/badge";

let mermaidId = 0;
function MermaidDiagram({ code }: { code: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const id = React.useMemo(() => `mermaid-svg-${mermaidId++}`, []);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(typeof window !== "undefined");
  }, []);

  useEffect(() => {
    if (!isClient || !ref.current) return;
    const diagramCode = code.trim() || "graph TD;A-->B;";
    console.log("Rendering Mermaid diagram with code:\n", diagramCode);
    (async () => {
      try {
        mermaid.initialize({ startOnLoad: false, theme: "neutral" });
        const { svg } = await mermaid.render(id, diagramCode);
        if (ref.current) ref.current.innerHTML = svg;
      } catch (err) {
        if (ref.current) ref.current.innerHTML = `<pre>${err}</pre>`;
      }
    })();
  }, [code, id, isClient]);

  if (!isClient) return <div className="my-4 text-gray-400">[Mermaid diagram will render on client]</div>;
  return <div ref={ref} className="my-4" />;
}

interface SectionContentProps {
  content: string;
  metadata?: {
    title?: string;
    description?: string;
    tags?: string[];
    section_id?: string;
    generated_at?: string;
    [key: string]: any;
  };
}

interface DebugState {
  originalContentType: string;
  originalContentLength: number;
  originalContentFirstChars: string;
  hasFrontmatter: boolean;
  frontmatterType: string | null;
  contentAfterProcessing: string;
}

export default function SectionContent({ content, metadata = {} }: SectionContentProps) {
  // The content should already be clean from the backend, but we'll do a final check
  const [processedContent, setProcessedContent] = useState("");
  const [debugState, setDebugState] = useState<DebugState>({
    originalContentType: typeof content,
    originalContentLength: content ? content.length : 0,
    originalContentFirstChars: content ? content.substring(0, 50) : "",
    hasFrontmatter: false,
    frontmatterType: null,
    contentAfterProcessing: ""
  });
  
  useEffect(() => {
    // Function to ensure content is clean
    const ensureCleanContent = (text: string): string => {
      console.log("[DEBUG SectionContent] Processing content:", {
        type: typeof text,
        length: text ? text.length : 0,
        startsWith: text ? text.substring(0, 50) : ""
      });
      
      if (!text) return "";
      
      let cleaned = text;
      let debug: DebugState = {
        originalContentType: typeof content,
        originalContentLength: content ? content.length : 0,
        originalContentFirstChars: content ? content.substring(0, 50) : "",
        hasFrontmatter: false,
        frontmatterType: null,
        contentAfterProcessing: ""
      };
      
      // Final clean-up for any remaining markdown code blocks or yaml blocks
      // This is just a safety check in case the backend missed something
      if (cleaned.startsWith('```yaml')) {
        const endIndex = cleaned.indexOf('\n```');
        if (endIndex > 0) {
          cleaned = cleaned.substring(endIndex + 4).trim();
          debug.hasFrontmatter = true;
          debug.frontmatterType = "```yaml (frontend cleanup)";
        }
      } else if (cleaned.startsWith('```markdown')) {
        const endIndex = cleaned.indexOf('\n```');
        if (endIndex > 0) {
          cleaned = cleaned.substring(endIndex + 4).trim();
          debug.hasFrontmatter = true;
          debug.frontmatterType = "```markdown (frontend cleanup)";
        }
      } else if (cleaned.startsWith('---')) {
        const endIndex = cleaned.indexOf('---', 3);
        if (endIndex > 0) {
          cleaned = cleaned.substring(endIndex + 3).trim();
          debug.hasFrontmatter = true;
          debug.frontmatterType = "--- (frontend cleanup)";
        }
      }
      
      // Check for JSON wrapped content (a common issue with nested quotes)
      if (cleaned.startsWith('{"content":')) {
        try {
          const parsed = JSON.parse(cleaned);
          if (parsed.content) {
            cleaned = parsed.content;
            debug.hasFrontmatter = true;
            debug.frontmatterType = "JSON wrapped (frontend cleanup)";
          }
        } catch (e) {
          console.log("[DEBUG SectionContent] Not valid JSON:", e);
        }
      }
      
      debug.contentAfterProcessing = cleaned.substring(0, 50);
      setDebugState(debug);
      
      console.log("[DEBUG SectionContent] Final content:", {
        length: cleaned.length,
        startsWith: cleaned.substring(0, 50)
      });
      
      return cleaned;
    };
    
    setProcessedContent(ensureCleanContent(content));
  }, [content]);
  
  return (
    <Card className="p-6">
      {/* Debug information */}
      {process.env.NODE_ENV !== 'production' && (
        <div className="mb-4 p-2 border border-gray-200 rounded bg-gray-50 text-xs">
          <details>
            <summary className="font-bold cursor-pointer">Debug Content Processing</summary>
            <pre className="mt-2 overflow-auto">
              {JSON.stringify(debugState, null, 2)}
            </pre>
          </details>
        </div>
      )}
    
      {metadata.tags && metadata.tags.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2">
          {metadata.tags.map((tag, index) => (
            <Badge key={index} variant="outline">{tag}</Badge>
          ))}
        </div>
      )}
      
      {metadata.description && (
        <div className="mb-4 text-sm text-gray-500 italic">
          {metadata.description}
        </div>
      )}
      
      <ReactMarkdown
        components={{
          code({ node, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            if (match && match[1] === "mermaid") {
              return <MermaidDiagram code={String(children).trim()} />;
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {processedContent}
      </ReactMarkdown>
      
      {metadata.generated_at && (
        <div className="mt-6 text-xs text-gray-400 text-right">
          Generated: {new Date(metadata.generated_at).toLocaleString()}
        </div>
      )}
    </Card>
  );
} 