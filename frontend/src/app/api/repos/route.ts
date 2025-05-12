import { NextResponse } from 'next/server';

type Repo = { 
  id: string; 
  name: string; 
  path: string;
  status?: string;
  has_structure?: boolean;
};

export async function GET() {
  let repos: Repo[] = [];
  try {
    // Call backend API instead of reading from filesystem
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
    const apiUrl = `${backendUrl}/list-wikis`;
    
    console.log(`[FRONTEND DEBUG] Fetching wikis from ${apiUrl}`);
    
    const res = await fetch(apiUrl);
    
    if (res.ok) {
      const data = await res.json();
      repos = data.wikis || [];
      
      console.log(`[FRONTEND DEBUG] Received ${repos.length} wikis from backend`);
      repos.forEach((repo, i) => {
        console.log(`[FRONTEND DEBUG]   Repo ${i+1}: id='${repo.id}', name='${repo.name}', status='${repo.status}'`);
      });
    } else {
      console.error('Failed to fetch wikis from backend:', await res.text());
    }
  } catch (err) {
    console.error('Error fetching wikis:', err);
    // fallback to empty list
  }
  return NextResponse.json({ repos });
} 