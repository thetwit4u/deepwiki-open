import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import MermaidDiagram from "./MermaidDiagram";

interface MarkdownRendererProps {
  content: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => (
  <div className="prose prose-sm max-w-none dark:prose-invert">
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeRaw]}
      components={{
        code({node, className, children, ...props}) {
          const match = /language-(\w+)/.exec(className || "");
          if (match && match[1] === "mermaid") {
            return <MermaidDiagram code={String(children).trim()} />;
          }
          return (
            <code className={className} {...props}>{children}</code>
          );
        },
        a: ({node, href, children, ...props}) => (
          <a href={href} className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer" {...props}>{children}</a>
        ),
        text({node, children, ...props}) {
          // Replace \n with <br /> in text nodes
          return <>{String(children).split("\n").map((line, i, arr) => (
            <React.Fragment key={i}>
              {line}
              {i < arr.length - 1 && <br />}
            </React.Fragment>
          ))}</>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  </div>
);

export default MarkdownRenderer; 