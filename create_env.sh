#!/bin/bash
# Create .env file in main folder

cat > .env << 'EOF'
# AI Port Decision-Support System - Environment Configuration
# Main project .env file - used by all submodules

# OpenAI Configuration
OPENAI_API_KEY=$OPENAI_API_KEY

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./RAG-LLM-for-Ports-backend/storage/chroma
CHROMA_COLLECTION_NAME=port_documents

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging Configuration
LOG_LEVEL=INFO

# Data Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENTS_PER_QUERY=10
EOF

echo "✅ .env file created in main folder"

