#!/bin/bash
# Test script to directly interact with the chat API using curl

COLLECTION_NAME="local_customs_exchange_rate_main_9cfa74b61a"
QUERY="What is this repository about and what are its key components?"

echo "Testing chat API with direct collection name"
echo "Collection: $COLLECTION_NAME"
echo "Query: $QUERY"
echo "---------------------------------"

curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d "{\"repo_id\": \"customs_exchange_rate_main\", \"message\": \"$QUERY\", \"generator_provider\": \"gemini\", \"embedding_provider\": \"ollama_nomic\", \"top_k\": 10, \"collection_name\": \"$COLLECTION_NAME\"}"

echo
echo "---------------------------------" 