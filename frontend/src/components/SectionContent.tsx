import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import mermaid from "mermaid";
import { Card } from "@/components/ui/card";

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

export default function SectionContent({ content }: { content: string }) {
  return (
    <Card className="p-6">
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
        {content}
      </ReactMarkdown>
    </Card>
  );
} 