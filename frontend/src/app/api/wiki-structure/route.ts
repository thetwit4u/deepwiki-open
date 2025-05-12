import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const repo = req.nextUrl.searchParams.get('repo');
  
  console.log(`[FRONTEND DEBUG] wiki-structure request with repo='${repo}'`);
  
  if (!repo) {
    console.log(`[FRONTEND DEBUG] Missing repo parameter`);
    return NextResponse.json({ error: 'Missing repo' }, { status: 400 });
  }
  
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
  const apiUrl = `${backendUrl}/wiki-structure?repo_url=${encodeURIComponent(repo)}`;
  
  console.log(`[FRONTEND DEBUG] Calling backend at: ${apiUrl}`);
  
  const res = await fetch(apiUrl);
  
  if (!res.ok) {
    console.log(`[FRONTEND DEBUG] Backend returned error: ${res.status} ${res.statusText}`);
    return NextResponse.json({ error: `Backend error: ${res.status} ${res.statusText}` }, { status: 500 });
  }
  
  const data = await res.json();
  console.log(`[FRONTEND DEBUG] Received structure with ${data.sections?.length || 0} sections`);
  
  return NextResponse.json(data);
} 