import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { repo_url, section_id, content, metadata } = body;
    
    console.log(`[API] Updating section content for ${section_id} in ${repo_url}`);
    console.log(`[API] Content length: ${content.length} characters`);
    
    if (!repo_url || !section_id || !content) {
      return NextResponse.json(
        { error: 'Missing required fields: repo_url, section_id, or content' },
        { status: 400 }
      );
    }
    
    // Forward the request to the Python backend
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
    const backendEndpoint = `${backendUrl}/update-section-content`;
    
    console.log(`[API] Forwarding request to backend at ${backendEndpoint}`);
    
    const backendResponse = await fetch(backendEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_url,
        section_id,
        content,
        metadata
      }),
    });
    
    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error(`[API] Backend error: ${backendResponse.status} - ${errorText}`);
      return NextResponse.json(
        { error: `Backend error: ${backendResponse.statusText}` },
        { status: backendResponse.status }
      );
    }
    
    const responseData = await backendResponse.json();
    console.log(`[API] Backend response:`, responseData);
    
    return NextResponse.json(responseData);
  } catch (error) {
    console.error('[API] Error updating section content:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
} 