import { NextRequest, NextResponse } from 'next/server';

const globalAny = global as any;
if (!globalAny.repoJobs) globalAny.repoJobs = {};

// Helper function to fetch with timeout
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

export async function GET(req: NextRequest) {
  const repo = req.nextUrl.searchParams.get('repo');
  if (!repo) {
    return NextResponse.json({ error: 'Missing repo' }, { status: 400 });
  }
  
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
  const progressUrl = `${backendUrl}/wiki-progress?repo_url=${encodeURIComponent(repo)}`;
  
  try {
    // Use a 2 second timeout to prevent blocking
    const res = await fetchWithTimeout(progressUrl, {}, 2000);
    
    if (!res.ok) {
      return NextResponse.json({ 
        status: 'error', 
        error: `Backend returned status ${res.status}`,
        log: [`Error fetching progress: ${res.statusText}`]
      });
    }
    
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error fetching progress:', error);
    
    // Handle abort errors (timeout)
    if (error.name === 'AbortError') {
      return NextResponse.json({ 
        status: 'processing', 
        log: ['Progress request timed out - still processing...'],
        error: 'Request timed out'
      });
    }
    
    // Handle other errors
    return NextResponse.json({ 
      status: 'error', 
      log: [`Error fetching progress: ${error.message || 'Unknown error'}`],
      error: error.message || 'Failed to fetch progress'
    });
  }
} 