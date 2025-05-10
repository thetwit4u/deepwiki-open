import { NextRequest, NextResponse } from 'next/server';

const globalAny = global as any;
if (!globalAny.repoJobs) globalAny.repoJobs = {};

export async function GET(req: NextRequest) {
  const repo = req.nextUrl.searchParams.get('repo');
  if (!repo) {
    return NextResponse.json({ error: 'Missing repo' }, { status: 400 });
  }
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
  const res = await fetch(`${backendUrl}/wiki-progress?repo_url=${encodeURIComponent(repo)}`);
  const data = await res.json();
  return NextResponse.json(data);
} 