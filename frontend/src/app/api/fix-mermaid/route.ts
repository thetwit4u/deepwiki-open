import { NextRequest, NextResponse } from 'next/server';

interface MermaidFixRequest {
  diagram: string;
  error: string;
  attempt: number;
}

interface MermaidFixResponse {
  fixed_diagram: string;
  changes_made?: string[];
  debug_info?: any;
}

export async function POST(req: NextRequest) {
  try {
    // Get request body
    const body = await req.json() as MermaidFixRequest;
    
    console.log('[FRONTEND DEBUG] fix-mermaid request:', {
      diagramLength: body.diagram?.length,
      errorLength: body.error?.length,
      attempt: body.attempt
    });
    
    if (!body.diagram || !body.error) {
      console.log('[FRONTEND DEBUG] Missing required fields in request');
      return NextResponse.json(
        { 
          error: 'Missing required fields: diagram and error are required',
          status: 'validation_error'
        }, 
        { status: 400 }
      );
    }
    
    // Set backend URL
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
    const apiUrl = `${backendUrl}/fix-mermaid`;
    
    console.log(`[FRONTEND DEBUG] Calling backend at: ${apiUrl}, attempt: ${body.attempt || 1}`);
    
    // Forward request to backend with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000); // 20 second timeout
    
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.log(`[FRONTEND DEBUG] Backend returned error: ${response.status} ${response.statusText}`);
        let errorDetail = 'Unknown error';
        
        try {
          // Try to parse error as JSON first
          const errorJson = await response.json();
          errorDetail = errorJson.detail || errorJson.error || JSON.stringify(errorJson);
        } catch {
          // If not JSON, get as text
          errorDetail = await response.text();
        }
        
        return NextResponse.json(
          { 
            error: `Backend error: ${errorDetail}`,
            status: 'backend_error',
            statusCode: response.status
          }, 
          { status: response.status }
        );
      }
      
      // Get response from backend
      const data = await response.json() as MermaidFixResponse;
      
      console.log(`[FRONTEND DEBUG] Received response from backend:`, {
        fixedDiagramLength: data.fixed_diagram?.length,
        changesCount: data.changes_made?.length || 0
      });
      
      // Enhanced response with additional metadata
      return NextResponse.json({
        fixed_diagram: data.fixed_diagram,
        changes_made: data.changes_made || [],
        attempt: body.attempt || 1,
        status: 'success',
        timestamp: new Date().toISOString()
      });
    } catch (fetchError) {
      clearTimeout(timeoutId);
      
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        console.error('[FRONTEND DEBUG] Request timed out');
        return NextResponse.json(
          { 
            error: 'Request timed out while waiting for diagram fix',
            status: 'timeout_error'
          }, 
          { status: 504 }
        );
      }
      
      throw fetchError; // Re-throw to be caught by outer catch
    }
  } catch (error) {
    console.error('[FRONTEND DEBUG] Error in fix-mermaid API route:', error);
    return NextResponse.json(
      { 
        error: `Internal server error: ${error instanceof Error ? error.message : String(error)}`,
        status: 'internal_error'
      }, 
      { status: 500 }
    );
  }
} 