import { NextRequest, NextResponse } from 'next/server';

const mockContent: Record<string, string> = {
  intro: `# Introduction\n\nWelcome to the DeepWiki for this repository. Here you'll find documentation, guides, and technical overviews.`,
  setup: `# Setup\n\nTo get started, clone the repo and run:\n\n\`\`\`bash\ngit clone https://github.com/example/repo.git\ncd repo\nnpm install\n\`\`\`\n`,
  usage: `# Usage\n\nImport the main module and call the entrypoint function.`,
  architecture: `# Architecture\n\nThis project uses a modular architecture with the following components:\n- API\n- Database\n- Frontend\n- Worker\n\nHere is a sample system diagram:\n\n\`\`\`mermaid\ngraph TD;\n  API-->Database;\n  API-->Frontend;\n  Frontend-->Worker;\n\`\`\`\n`,
  api: `# API Reference\n\nSee the OpenAPI spec in \`openapi.yaml\`.`,
  faq: `# FAQ\n\n**Q:** How do I contribute?\n**A:** Fork the repo and submit a PR.`,
};

export async function GET(req: NextRequest) {
  const repo = req.nextUrl.searchParams.get('repo');
  const section = req.nextUrl.searchParams.get('section');
  if (!repo || !section) {
    return NextResponse.json({ content: '# Section\n\nContent not found.' }, { status: 400 });
  }
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
  const res = await fetch(`${backendUrl}/wiki-section?repo_url=${encodeURIComponent(repo)}&section_id=${encodeURIComponent(section)}`);
  if (!res.ok) {
    return NextResponse.json({ content: '# Section\n\nContent not found.' }, { status: 404 });
  }
  const content = await res.text();
  return NextResponse.json({ content });
} 