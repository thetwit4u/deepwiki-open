import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(req: NextRequest) {
  console.log('[API ROUTE DEBUG] section-content/update: Request received');
  try {
    const body = await req.json();
    
    console.log('[API ROUTE DEBUG] section-content/update: Request body parsed');
    console.log('[API ROUTE DEBUG] section-content/update: repo_url =', body.repo_url);
    console.log('[API ROUTE DEBUG] section-content/update: section_id =', body.section_id);
    console.log('[API ROUTE DEBUG] section-content/update: content length =', body.content?.length || 0);
    
    // Validate required fields
    if (!body.repo_url || !body.section_id || !body.content) {
      console.log('[API ROUTE DEBUG] section-content/update: Missing required fields:', { 
        repo_url: !!body.repo_url, 
        section_id: !!body.section_id, 
        content: !!body.content 
      });
      return NextResponse.json(
        { error: 'Missing required fields: repo_url, section_id, and content are required' },
        { status: 400 }
      );
    }
    
    // Backend API URL
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8001';
    const apiUrl = `${backendUrl}/update-section-content`;
    
    console.log(`[API ROUTE DEBUG] section-content/update: Calling backend at: ${apiUrl}`);
    
    // Forward the request to the backend
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_url: body.repo_url,
        section_id: body.section_id,
        content: body.content,
        metadata: body.metadata || {}
      }),
    });
    
    console.log(`[API ROUTE DEBUG] section-content/update: Backend response status: ${response.status}`);
    
    if (!response.ok) {
      console.log(`[API ROUTE DEBUG] section-content/update: Backend returned error: ${response.status} ${response.statusText}`);
      let errorText = "";
      try {
        errorText = await response.text();
        console.log(`[API ROUTE DEBUG] section-content/update: Error details: ${errorText}`);
      } catch (readErr) {
        console.log(`[API ROUTE DEBUG] section-content/update: Could not read error details: ${readErr}`);
      }
      
      return NextResponse.json(
        { error: `Backend error: ${errorText || response.statusText}` },
        { status: response.status }
      );
    }
    
    // Get the response from the backend
    let data;
    try {
      data = await response.json();
      console.log(`[API ROUTE DEBUG] section-content/update: Backend response data:`, JSON.stringify(data).substring(0, 200) + '...');
    } catch (jsonErr) {
      console.log(`[API ROUTE DEBUG] section-content/update: Error parsing backend response: ${jsonErr}`);
      data = {};
    }
    
    // Return success with the updated content details
    console.log(`[API ROUTE DEBUG] section-content/update: Returning success response`);
    return NextResponse.json({
      success: true,
      message: 'Content updated successfully',
      section_id: body.section_id,
      ...data
    });
  } catch (error) {
    console.error('[API ROUTE DEBUG] section-content/update: Error:', error);
    return NextResponse.json(
      { 
        error: `Internal server error: ${error instanceof Error ? error.message : String(error)}`,
        success: false
      },
      { status: 500 }
    );
  }
} 