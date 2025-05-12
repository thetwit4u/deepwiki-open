import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    // Get request body
    const body = await req.json();
    
    console.log('[FRONTEND DEBUG] fix-mermaid request:', body);
    
    if (!body.diagram || !body.error) {
      console.log('[FRONTEND DEBUG] Missing required fields in request');
      return NextResponse.json(
        { error: 'Missing required fields: diagram and error are required' }, 
        { status: 400 }
      );
    }
    
    // Set backend URL
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
    const apiUrl = `${backendUrl}/fix-mermaid`;
    
    console.log(`[FRONTEND DEBUG] Calling backend at: ${apiUrl}`);
    
    // Forward request to backend
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    if (!response.ok) {
      console.log(`[FRONTEND DEBUG] Backend returned error: ${response.status} ${response.statusText}`);
      const errorText = await response.text();
      return NextResponse.json(
        { error: `Backend error: ${errorText}` }, 
        { status: response.status }
      );
    }
    
    // Get response from backend
    const data = await response.json();
    console.log(`[FRONTEND DEBUG] Received response from backend:`, data);
    
    // Return the fixed diagram
    return NextResponse.json(data);
  } catch (error) {
    console.error('[FRONTEND DEBUG] Error in fix-mermaid API route:', error);
    return NextResponse.json(
      { error: `Internal server error: ${error instanceof Error ? error.message : String(error)}` }, 
      { status: 500 }
    );
  }
} 