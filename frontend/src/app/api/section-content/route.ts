import { NextRequest, NextResponse } from 'next/server';
import yaml from 'js-yaml';

const mockContent: Record<string, string> = {
  intro: `# Introduction\n\nWelcome to the DeepWiki for this repository. Here you'll find documentation, guides, and technical overviews.`,
  setup: `# Setup\n\nTo get started, clone the repo and run:\n\n\`\`\`bash\ngit clone https://github.com/example/repo.git\ncd repo\nnpm install\n\`\`\`\n`,
  usage: `# Usage\n\nImport the main module and call the entrypoint function.`,
  architecture: `# Architecture\n\nThis project uses a modular architecture with the following components:\n- API\n- Database\n- Frontend\n- Worker\n\nHere is a sample system diagram:\n\n\`\`\`mermaid\ngraph TD;\n  API-->Database;\n  API-->Frontend;\n  Frontend-->Worker;\n\`\`\`\n`,
  api: `# API Reference\n\nSee the OpenAPI spec in \`openapi.yaml\`.`,
  faq: `# FAQ\n\n**Q:** How do I contribute?\n**A:** Fork the repo and submit a PR.`,
};

// Function to extract YAML frontmatter
function extractFrontmatter(content: string): { metadata: any; content: string } {
  // Check if content starts with ```yaml or ---
  const yamlPattern = /^```yaml\s*\n([\s\S]*?)```\s*\n/;
  const dashPattern = /^---\s*\n([\s\S]*?)---\s*\n/;
  
  let cleanedContent = content;
  let metadata = {};
  
  try {
    // Extract YAML frontmatter if it exists
    if (yamlPattern.test(content)) {
      const match = content.match(yamlPattern);
      if (match && match[1]) {
        metadata = yaml.load(match[1]) || {};
        cleanedContent = content.replace(yamlPattern, '');
      }
    } else if (dashPattern.test(content)) {
      const match = content.match(dashPattern);
      if (match && match[1]) {
        metadata = yaml.load(match[1]) || {};
        cleanedContent = content.replace(dashPattern, '');
      }
    }
  } catch (error) {
    console.error('Error parsing frontmatter:', error);
  }
  
  return {
    metadata,
    content: cleanedContent.trim()
  };
}

// Function to unescape JSON string
function unescapeJsonString(str: string): string {
  // Replace escaped sequences with their actual characters
  return str
    .replace(/\\"/g, '"')
    .replace(/\\n/g, '\n')
    .replace(/\\r/g, '\r')
    .replace(/\\t/g, '\t')
    .replace(/\\\\/g, '\\')
    .replace(/\\`/g, '`');
}

export async function GET(req: NextRequest) {
  const repo = req.nextUrl.searchParams.get('repo');
  const section = req.nextUrl.searchParams.get('section');
  
  console.log(`[FRONTEND DEBUG] section-content request with repo='${repo}', section='${section}'`);
  
  if (!repo || !section) {
    console.log(`[FRONTEND DEBUG] Missing parameters: repo=${repo}, section=${section}`);
    return NextResponse.json({ content: '# Section\n\nContent not found.', metadata: {} }, { status: 400 });
  }
  
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
  const apiUrl = `${backendUrl}/wiki-section?repo_url=${encodeURIComponent(repo)}&section_id=${encodeURIComponent(section)}`;
  
  console.log(`[FRONTEND DEBUG] Calling backend at: ${apiUrl}`);
  
  const res = await fetch(apiUrl);
  
  if (!res.ok) {
    console.log(`[FRONTEND DEBUG] Backend returned error: ${res.status} ${res.statusText}`);
    return NextResponse.json({ content: '# Section\n\nContent not found.', metadata: {} }, { status: 404 });
  }
  
  let rawContent = await res.text();
  console.log(`[FRONTEND DEBUG] Received ${rawContent.length} bytes of content`);
  
  // Debug the raw content structure
  console.log(`[FRONTEND DEBUG] Content starts with: ${rawContent.substring(0, 100)}`);
  
  // Check if content appears to be nested JSON (JSON string within JSON)
  let parsedData: any = {};
  try {
    // First parse the outer JSON
    parsedData = JSON.parse(rawContent);
    console.log(`[FRONTEND DEBUG] Parsed outer JSON. Content field type: ${typeof parsedData.content}`);
    
    // Check if content looks like a stringified JSON
    if (typeof parsedData.content === 'string' && 
        (parsedData.content.startsWith('{') || parsedData.content.startsWith('{"'))) {
      try {
        // Try to parse the inner JSON
        const innerJson = JSON.parse(parsedData.content);
        console.log(`[FRONTEND DEBUG] Found nested JSON. Unwrapping.`);
        
        // If inner JSON has content and metadata, use those directly
        if (innerJson.content) {
          return NextResponse.json({
            content: innerJson.content,
            metadata: innerJson.metadata || {}
          });
        }
      } catch (innerErr) {
        console.log(`[FRONTEND DEBUG] Failed to parse inner JSON: ${innerErr}`);
        // Continue with regular flow if inner JSON parsing fails
      }
    }
    
    // Check if content looks like it's JSON-escaped (has \\n, \\\`, etc.)
    if (parsedData.content && 
        (parsedData.content.includes('\\n') || 
         parsedData.content.includes('\\`') || 
         parsedData.content.includes('\\\\'))) {
      parsedData.content = unescapeJsonString(parsedData.content);
    }
    
    // Extract frontmatter if present
    if (parsedData.content) {
      const { metadata, content } = extractFrontmatter(parsedData.content);
      return NextResponse.json({
        content: content,
        metadata: { ...parsedData.metadata, ...metadata }
      });
    }
    
    return NextResponse.json(parsedData);
  } catch (err) {
    console.error(`[FRONTEND DEBUG] Error parsing response: ${err}`, rawContent);
    
    // Fall back to returning the raw content if JSON parsing fails
    return NextResponse.json({ 
      content: rawContent,
      metadata: {}
    });
  }
} 