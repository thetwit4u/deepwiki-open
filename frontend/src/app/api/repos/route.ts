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
    const res = await fetch(`${backendUrl}/list-wikis`);
    
    if (res.ok) {
      const data = await res.json();
      repos = data.wikis || [];
    } else {
      console.error('Failed to fetch wikis from backend:', await res.text());
    }
  } catch (err) {
    console.error('Error fetching wikis:', err);
    // fallback to empty list
  }
  return NextResponse.json({ repos });
} 