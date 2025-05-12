import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    // Get request body
    const body = await req.json();
    
    console.log('[FRONTEND DEBUG] chat request:', body);
    
    if (!body.repoId || !body.message) {
      console.log('[FRONTEND DEBUG] Missing required fields in request');
      return NextResponse.json(
        { error: 'Missing required fields: repoId and message are required' }, 
        { status: 400 }
      );
    }
    
    // Set backend URL
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
    const apiUrl = `${backendUrl}/chat`;
    
    console.log(`[FRONTEND DEBUG] Calling backend at: ${apiUrl}`);
    
    // Forward request to backend
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_id: body.repoId,
        message: body.message,
        generator_provider: body.generatorProvider || 'gemini',
        embedding_provider: body.embeddingProvider || 'openai',
        top_k: body.topK || 10,
      }),
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
    console.log(`[FRONTEND DEBUG] Received response from backend with ${
      data.retrieved_documents?.length || 0
    } documents`);
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('[FRONTEND DEBUG] Error in chat API route:', error);
    return NextResponse.json(
      { error: `Internal server error: ${error instanceof Error ? error.message : String(error)}` }, 
      { status: 500 }
    );
  }
}

export async function GET(req: NextRequest) {
  try {
    const repoId = req.nextUrl.searchParams.get('repoId');
    
    if (!repoId) {
      return NextResponse.json(
        { error: 'Missing required parameter: repoId' }, 
        { status: 400 }
      );
    }
    
    // Set backend URL
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
    const apiUrl = `${backendUrl}/chat/history?repo_id=${encodeURIComponent(repoId)}`;
    
    console.log(`[FRONTEND DEBUG] Fetching chat history from: ${apiUrl}`);
    
    // Get chat history from backend
    const response = await fetch(apiUrl);
    
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
    console.log(`[FRONTEND DEBUG] Received chat history with ${data.messages?.length || 0} messages`);
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('[FRONTEND DEBUG] Error in chat history API route:', error);
    return NextResponse.json(
      { error: `Internal server error: ${error instanceof Error ? error.message : String(error)}` }, 
      { status: 500 }
    );
  }
} 