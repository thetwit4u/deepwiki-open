#!/usr/bin/env node
/**
 * Test script to directly test the frontend chat API with a custom collection name
 */

const fetch = require('node-fetch');

// Configuration
const API_URL = 'http://localhost:3000/api/chat';
const REPO_ID = 'customs_exchange_rate_main';
const COLLECTION_NAME = 'local_customs_exchange_rate_main_9cfa74b61a';
const QUERY = 'What is this repository about and what are its key components?';

async function testChat() {
  console.log('Testing frontend chat API with custom collection name');
  console.log(`Repository ID: ${REPO_ID}`);
  console.log(`Collection name: ${COLLECTION_NAME}`);
  console.log(`Query: ${QUERY}`);
  console.log('-'.repeat(60));

  try {
    // Create request payload
    const payload = {
      repoId: REPO_ID,
      message: QUERY,
      generatorProvider: 'gemini',
      embeddingProvider: 'ollama_nomic',
      topK: 10,
      collectionName: COLLECTION_NAME  // Pass the collection name to frontend
    };

    console.log('Sending request with payload:', JSON.stringify(payload, null, 2));

    // Send request to API
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    // Parse response
    const data = await response.json();

    if (data.error) {
      console.error('Error from API:', data.error);
      process.exit(1);
    }

    // Print the response
    console.log('\n=== ANSWER ===');
    console.log(data.answer || 'No answer received');

    // Print metadata (excluding retrieved documents for brevity)
    if (data.metadata) {
      console.log('\n=== METADATA ===');
      const { retrieved_documents, ...restMetadata } = data.metadata;
      console.log(JSON.stringify(restMetadata, null, 2));
      
      if (retrieved_documents) {
        console.log(`\nNumber of retrieved documents: ${retrieved_documents.length}`);
      }
    }

    console.log('\n✅ Test completed successfully');
  } catch (error) {
    console.error('Error during test:', error.message);
    process.exit(1);
  }
}

// Run the test
testChat(); 